#include "agent.cpp"
#include "rollout_buffer.cpp"
#include "../environment/stratego_env.cpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <torch/torch.h>

// Define the Actor-Critic Neural Network Architecture
struct StrategoNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear shared_fc{nullptr}; // Change: Shared FC like Python
    torch::nn::Linear actor_head{nullptr};
    torch::nn::Linear critic_head{nullptr};

    StrategoNetImpl(int channels, int width, int height, int action_dim) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 64, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        
        int flat_size = 128 * width * height;
        shared_fc = register_module("shared_fc", torch::nn::Linear(flat_size, 256));
        actor_head = register_module("actor_head", torch::nn::Linear(256, action_dim));
        critic_head = register_module("critic_head", torch::nn::Linear(256, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = x.reshape({x.size(0), -1});
        
        torch::Tensor features = torch::relu(shared_fc(x));
        
        // Add Tanh back to the critic to match Python's output range [-1, 1]
        return {actor_head(features), torch::tanh(critic_head(features))};
    }
};
TORCH_MODULE(StrategoNet);


class StrategoTinyAgent : public Agent<StrategoObs, StrategoAction> {
private:
    const StrategoEnvironment& env;
    StrategoNet net;
    torch::optim::Adam optimizer;
    bool eval_mode = false;

public:
    StrategoTinyAgent(const StrategoEnvironment& environment) 
        : env(environment), 
          net(9, environment.get_board().get_width(), environment.get_board().get_height(), environment.action_dim()),
          optimizer(net->parameters(), torch::optim::AdamOptions(1e-4)) {
    }

    void set_eval_mode(bool eval) {
        eval_mode = eval;
        if (eval) net->eval();
        else net->train();
    }

    void save_model(const std::string& path) const {
        torch::serialize::OutputArchive archive;
        net->save(archive);
        archive.save_to(path);
    }

    void load_model(const std::string& path) {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        net->load(archive);
    }

    void copy_weights_from(const StrategoTinyAgent& other) {
        torch::autograd::GradMode::set_enabled(false);
        for (auto& val : other.net->named_parameters()) net->named_parameters()[val.key()].copy_(val.value());
        for (auto& val : other.net->named_buffers()) net->named_buffers()[val.key()].copy_(val.value());
        torch::autograd::GradMode::set_enabled(true);
    }

    [[nodiscard]] AgentOutput<StrategoAction> act(const StrategoObs& obs) override {
        torch::NoGradGuard no_grad; // Disable gradients during rollout
        
        std::vector<StrategoAction> legal_actions = env.get_legal_actions();

        int w = env.get_board().get_width();
        int h = env.get_board().get_height();
        
        if (legal_actions.empty()) {
            return {0, 0.0f, 0.0f, std::vector<float>(env.action_dim(), -1e9f)}; // Fallback if no moves are available
        }

        // --- UPDATE 1: READ DIRECTLY AS CHW ---
        // Since encode_board now provides CHW, we map it directly.
        // Shape: {Batch=1, Channels=9, Height, Width}
        torch::Tensor obs_tensor = torch::from_blob((void*)obs.data(), 
                                   {1, 9, (long)env.get_board().get_height(), (long)env.get_board().get_width()}, 
                                   torch::kFloat32).clone();
        
        auto [logits, value] = net->forward(obs_tensor);
        
        // Action Masking: Set illegal actions to -infinity
        torch::Tensor mask = torch::full({1, (long)env.action_dim()}, -1e9, torch::kFloat32);
        for (StrategoAction a : legal_actions) {
            mask[0][a] = 0.0f;
        }
        
        logits = logits + mask;

        // Sample from the probability distribution
        torch::Tensor probs = torch::softmax(logits, /*dim=*/-1);
        StrategoAction chosen_action;
        
        if (!eval_mode) {
            torch::Tensor action_tensor = torch::multinomial(probs, 1);
            chosen_action = action_tensor.item<int>();
        } else {
            chosen_action = torch::argmax(probs).item<int>();
        }
        
        // Get Log Probability and Critic Value
        torch::Tensor log_probs = torch::log_softmax(logits, /*dim=*/-1);
        float log_prob = log_probs[0][chosen_action].item<float>();
        float val = value[0][0].item<float>();

        std::vector<float> mask_vec(mask.data_ptr<float>(), mask.data_ptr<float>() + mask.numel());

        return {chosen_action, log_prob, val, mask_vec};
    }

    void update_weights(RolloutBuffer<StrategoObs, StrategoAction>& buffer) override {
        // 0. Define Hyperparameters and Dimensions
        int ppo_epochs = 4;
        float clip_eps = 0.2f;
        float c1 = 0.5f;   // Critic loss
        float c2 = 0.01f;  // Entropy
        
        int w = env.get_board().get_width();
        int h = env.get_board().get_height();
        size_t batch_size = buffer.observations.size();
        if (batch_size == 0) return;

        // 1. Flatten Observations and Masks
        std::vector<float> flat_obs;
        std::vector<float> flat_masks;
        flat_obs.reserve(batch_size * 9 * w * h);
        flat_masks.reserve(batch_size * env.action_dim());

        for (const auto& obs : buffer.observations) {
            flat_obs.insert(flat_obs.end(), obs.begin(), obs.end());
        }
        for (const auto& m : buffer.masks) {
            flat_masks.insert(flat_masks.end(), m.begin(), m.end());
        }

        // 2. Create Tensors
        torch::Tensor obs_tensor = torch::from_blob(flat_obs.data(), {(long)batch_size, 9, (long)h, (long)w}, torch::kFloat32).clone();
        torch::Tensor masks_tensor = torch::from_blob(flat_masks.data(), {(long)batch_size, (long)env.action_dim()}, torch::kFloat32).clone();
        
        torch::Tensor old_actions = torch::from_blob(buffer.actions.data(), {(long)batch_size}, torch::kInt32).to(torch::kInt64).clone();
        torch::Tensor old_log_probs = torch::from_blob(buffer.log_probs.data(), {(long)batch_size}, torch::kFloat32).clone();
        torch::Tensor returns = torch::from_blob(buffer.returns.data(), {(long)batch_size}, torch::kFloat32).clone();
        torch::Tensor advantages = torch::from_blob(buffer.advantages.data(), {(long)batch_size}, torch::kFloat32).clone();

        // Normalization (matches Python)
        if (batch_size > 1 && returns.std().item<float>() > 1e-5) {
            returns = (returns - returns.mean()) / (returns.std() + 1e-5f);
        }
        if (batch_size > 1) {
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);
        }

        // 3. PPO Epoch Loop
        for (int epoch = 0; epoch < ppo_epochs; ++epoch) {
            auto [logits, values] = net->forward(obs_tensor);
            values = values.squeeze(-1);

            // Apply masks
            logits = logits + masks_tensor;
            
            torch::Tensor log_probs_dist = torch::log_softmax(logits, /*dim=*/-1);
            torch::Tensor probs_dist = torch::softmax(logits, /*dim=*/-1);

            // Extract log probabilities for the specific actions that were actually taken
            torch::Tensor new_log_probs = log_probs_dist.gather(1, old_actions.unsqueeze(1)).squeeze(1);
            
            // Entropy for exploration bonus, handling NaNs from 0 * -inf masks
            torch::Tensor entropy = torch::nan_to_num(-(probs_dist * log_probs_dist), 0.0).sum(-1);

            // Probability ratio: r_t(θ) = exp(new_log_prob - old_log_prob)
            torch::Tensor ratio = torch::exp(new_log_probs - old_log_probs);

            // Surrogate Loss (Clipped)
            torch::Tensor surr1 = ratio * advantages;
            torch::Tensor surr2 = torch::clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps) * advantages;
            torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
            torch::Tensor critic_loss = torch::nn::functional::mse_loss(values, returns);
            torch::Tensor total_loss = actor_loss + c1 * critic_loss - c2 * entropy.mean();

            // Backpropagation
            optimizer.zero_grad();
            total_loss.backward();
            optimizer.step();
        }
    }
};