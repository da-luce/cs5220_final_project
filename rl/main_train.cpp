#include "rl/environment/stratego_env.h"
#include "rl/agent/ppo_agent.h"
#include "rl/agent/random_policy.h"
#include "rl/policy_neural.h"
#include "networks/model.h"
#include "networks/torsos/cnn.h"
#include "rl/encoding/board.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <chrono>

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    stratego::BoardConfig config = stratego::get_config_for_game_type(stratego::GameType::Tiny);
    auto torso = std::make_shared<networks::torsos::CNNTorsoImpl>(9, 64, 0);
    networks::StrategoNet model(torso, 64, 4 * 4);
    model->to(torch::kCPU);

    // Python Parity settings
    PPOAgent agent(model, 1e-4, 0.99, 1, 0.2); 
    RolloutBuffer<torch::Tensor, int> buffer;
    auto env = std::make_unique<StrategoEnvironment>(config, stratego::state::SetupType::Random, 60);

    std::cout << "Starting PYTHON-ALIGNED Training..." << std::endl;

    for (int i = 0; i < 20000; ++i) {
        torch::Tensor obs = env->reset();
        bool done = false;
        std::vector<stratego::Player> move_owners;

        while (!done) {
            torch::Tensor mask = env->get_action_mask();
            if (mask.sum().item<float>() == 0) break;

            move_owners.push_back(env->get_current_player());
            auto output = agent.act(obs, mask);
            auto step_res = env->step(output.action);
            
            buffer.observations.push_back(obs);
            buffer.actions.push_back(output.action);
            buffer.log_probs.push_back(output.log_prob);
            buffer.values.push_back(output.value);
            buffer.masks.push_back(mask);
            buffer.rewards.push_back(0.0f);
            buffer.is_terminals.push_back(false);

            obs = step_res.observation;
            done = step_res.terminated;

            if (done) {
                float final_reward = step_res.reward;
                stratego::Player last_mover = move_owners.back();
                buffer.is_terminals.back() = true;

                int start_idx = (int)buffer.size() - (int)move_owners.size();
                for (int j = 0; j < (int)move_owners.size(); ++j) {
                    buffer.rewards[start_idx + j] = (move_owners[j] == last_mover) ? final_reward : -final_reward;
                }
            }
        }

        // Update every game to match Python exactly
        agent.update_weights(buffer);
        buffer.clear();

        if (i % 2000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            std::cout << "Episode " << i << " | Time: " << duration << "s" << std::endl;
        }
    }

    torch::save(model, "stratego_model.pt");
    std::cout << "Training complete. Model saved to stratego_model.pt" << std::endl;
    return 0;
}
