#include "ppo_agent.h"
#include <algorithm>
#include <cmath>

PPOAgent::PPOAgent(networks::StrategoNet model, 
                   double lr, 
                   double gamma, 
                   double k_epochs, 
                   double eps_clip)
    : model(model), 
      optimizer(model->parameters(), torch::optim::AdamOptions(lr)),
      gamma(gamma), 
      k_epochs(k_epochs), 
      eps_clip(eps_clip) {}

AgentOutput<int> PPOAgent::act(const torch::Tensor& obs, const torch::Tensor& mask) {
    torch::NoGradGuard no_grad;
    model->eval();

    auto device = model->parameters()[0].device();
    torch::Tensor input = obs.to(device).unsqueeze(0);
    auto [logits, value] = model->forward(input);
    
    torch::Tensor flat_logits = logits.view({1, -1});
    // Apply mask in log-space
    torch::Tensor masked_logits = flat_logits.clone();
    masked_logits.masked_fill_(mask.to(device).unsqueeze(0) == 0, -1e9);
    
    torch::Tensor probs = torch::softmax(masked_logits, 1);
    
    // Stable sampling
    torch::Tensor action_tensor = torch::multinomial(probs, 1);
    int action = action_tensor.item<int>();
    
    // Stable log_prob
    torch::Tensor log_probs = torch::log_softmax(masked_logits, 1);
    float log_prob = log_probs.index({0, action}).item<float>();
    
    AgentOutput<int> output;
    output.action = action;
    output.log_prob = log_prob;
    output.value = value.item<float>();
    
    return output;
}

BatchedAgentOutput PPOAgent::act_batch(const torch::Tensor& obs_batch, const torch::Tensor& mask_batch) {
    torch::NoGradGuard no_grad;
    model->eval();
    auto device = model->parameters()[0].device();
    int N = obs_batch.size(0);

    auto [logits, values] = model->forward(obs_batch.to(device));
    torch::Tensor flat_logits = logits.view({N, -1});
    torch::Tensor masked_logits = flat_logits.clone();
    masked_logits.masked_fill_(mask_batch.to(device) == 0, -1e9);

    torch::Tensor probs = torch::softmax(masked_logits, 1);
    torch::Tensor actions = torch::multinomial(probs, 1).squeeze(1);           // [N]
    torch::Tensor log_prob_mat = torch::log_softmax(masked_logits, 1);
    torch::Tensor sel_log_probs = log_prob_mat.gather(1, actions.unsqueeze(1)).squeeze(1); // [N]
    torch::Tensor vals = values.squeeze(1);                                    // [N]

    // One batch transfer to CPU instead of 3N individual GPU syncs
    auto act_cpu = actions.to(torch::kCPU);
    auto lp_cpu  = sel_log_probs.to(torch::kCPU);
    auto val_cpu = vals.to(torch::kCPU);

    auto* act_ptr = act_cpu.data_ptr<int64_t>();
    auto* lp_ptr  = lp_cpu.data_ptr<float>();
    auto* val_ptr = val_cpu.data_ptr<float>();

    BatchedAgentOutput out;
    out.actions.resize(N);
    out.log_probs.resize(N);
    out.values.resize(N);
    for (int i = 0; i < N; ++i) {
        out.actions[i]   = static_cast<int>(act_ptr[i]);
        out.log_probs[i] = lp_ptr[i];
        out.values[i]    = val_ptr[i];
    }
    return out;
}

void PPOAgent::update_weights(RolloutBuffer<torch::Tensor, int>& buffer) {
    if (buffer.size() == 0) return;
    model->train();

    auto device = model->parameters()[0].device();

    // 1. Discounted Returns
    std::vector<float> returns_vec(buffer.size());
    float discounted_reward = 0;
    for (int i = (int)buffer.size() - 1; i >= 0; --i) {
        if (buffer.is_terminals[i]) discounted_reward = 0;
        discounted_reward = buffer.rewards[i] + (gamma * discounted_reward);
        returns_vec[i] = discounted_reward;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor states = torch::stack(buffer.observations).to(device);
    torch::Tensor actions = torch::tensor(buffer.actions, torch::kInt64).to(device);
    torch::Tensor old_logprobs = torch::tensor(buffer.log_probs, options).to(device);
    torch::Tensor returns = torch::tensor(returns_vec, options).to(device);
    torch::Tensor old_values = torch::tensor(buffer.values, options).to(device);
    torch::Tensor masks = torch::stack(buffer.masks).view({(int)buffer.size(), -1}).to(device);

    // Python Parity: Normalize returns
    if (buffer.size() > 1) {
        returns = (returns - returns.mean()) / (returns.std() + 1e-5);
    }

    // 2. Optimization Loop
    for (int epoch = 0; epoch < k_epochs; ++epoch) {
        auto [logits, values] = model->forward(states);
        torch::Tensor flat_logits = logits.view({(int)buffer.size(), -1});
        
        // Stable Masking
        torch::Tensor masked_logits = flat_logits.clone();
        masked_logits.masked_fill_(masks == 0, -1e9);
        
        torch::Tensor log_probs = torch::log_softmax(masked_logits, 1);
        torch::Tensor probs = torch::softmax(masked_logits, 1);
        
        torch::Tensor new_logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1);
        torch::Tensor entropy = -(probs * log_probs).sum(1).mean();
        
        torch::Tensor ratios = torch::exp(new_logprobs - old_logprobs);
        // Advantages calculated using rollout values to match Python stability
        torch::Tensor advantages = returns - old_values; 
        
        torch::Tensor surr1 = ratios * advantages;
        torch::Tensor surr2 = torch::clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages;
        
        torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
        torch::Tensor critic_loss = torch::mse_loss(values.squeeze(1), returns);
        
        // Python parity: 0.01 entropy coeff
        torch::Tensor loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
