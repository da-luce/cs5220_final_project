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
#include <string>

struct EvalResult {
    float win_rate;
    float draw_rate;
};

EvalResult evaluate_vs_champion(
    networks::StrategoNet challenger,
    networks::StrategoNet champion,
    const stratego::BoardConfig& config,
    stratego::state::SetupType setup_type,
    int num_games = 50
) {
    challenger->eval();
    champion->eval();
    torch::NoGradGuard no_grad;

    int challenger_wins = 0;
    int champion_wins = 0;
    int draws = 0;

    for (int g = 0; g < num_games; ++g) {
        StrategoEnvironment env(config, setup_type, 60);
        bool challenger_is_red = (g % 2 == 0);
        bool done = false;
        torch::Tensor obs = env.reset();

        int moves_in_game = 0;
        while (!done) {
            torch::Tensor mask = env.get_action_mask();
            stratego::Player current_p = env.get_current_player();
            bool is_challenger_turn = (current_p == stratego::Player::Red && challenger_is_red) ||
                                      (current_p == stratego::Player::Blue && !challenger_is_red);

            if (mask.sum().item<float>() == 0) {
                // Current player is trapped. Opponent wins.
                if (is_challenger_turn) champion_wins++; 
                else challenger_wins++;
                done = true;
                break;
            }
            
            auto& active_net = is_challenger_turn ? challenger : champion;
            auto [logits, value] = active_net->forward(obs.unsqueeze(0));
            
            torch::Tensor masked_logits = logits.view({1, -1}).clone();
            masked_logits.masked_fill_(mask.unsqueeze(0) == 0, -1e9);

            int action = torch::argmax(masked_logits, 1).item<int>();
            auto step_res = env.step(action);
            obs = step_res.observation;
            done = step_res.terminated;
            moves_in_game++;

            if (done) {
                if (step_res.reward == 1.0f) {
                    // The player who just moved won.
                    if (is_challenger_turn) challenger_wins++;
                    else champion_wins++;
                } else if (step_res.reward == -1.0f) {
                    // The player who just moved made an invalid move and lost.
                    if (is_challenger_turn) champion_wins++;
                    else challenger_wins++;
                } else {
                    draws++;
                }
            }
        }
    }

    std::cout << "  [Eval Stats] Challenger: " << challenger_wins << " | Champion: " << champion_wins << " | Draws: " << draws << std::endl;

    challenger->train();
    return {(float)challenger_wins / num_games, (float)draws / num_games};
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string variant_str = "tiny";
    std::string setup_str = "random";
    int num_episodes = 20000;
    if (argc >= 2) variant_str = argv[1];
    if (argc >= 3) setup_str = argv[2];
    if (argc >= 4) num_episodes = std::stoi(argv[3]);

    stratego::GameType game_type;
    if (variant_str == "classic") game_type = stratego::GameType::Classic;
    else if (variant_str == "quick") game_type = stratego::GameType::Quick;
    else if (variant_str == "barrage") game_type = stratego::GameType::Barrage;
    else game_type = stratego::GameType::Tiny;

    stratego::state::SetupType setup_type;
    if (setup_str == "default") setup_type = stratego::state::SetupType::Default;
    else if (setup_str == "probabilistic") setup_type = stratego::state::SetupType::Probabilistic;
    else setup_type = stratego::state::SetupType::Random;

    stratego::BoardConfig config = stratego::get_config_for_game_type(game_type);
    
    std::string model_name = variant_str + "_" + setup_str + ".pt";
    std::string model_path = std::string(PROJECT_ROOT_DIR) + "/models/" + model_name;

    auto env = std::make_unique<StrategoEnvironment>(config, setup_type, 60);

    int obs_channels = get_encoding_channels(config);
    int action_channels = env->get_action_encoder().get_action_channels();
    int H = config.height;
    int W = config.width;

    std::cout << "Config: Obs Channels=" << obs_channels << " Action Channels=" << action_channels << " Dim=" << H << "x" << W << std::endl;
    std::cout << "Target Path: " << model_path << std::endl;

    auto torso_challenger = std::make_shared<networks::torsos::CNNTorsoImpl>(obs_channels, 64, 0);
    networks::StrategoNet challenger(torso_challenger, action_channels * H * W, H * W);
    
    auto torso_champion = std::make_shared<networks::torsos::CNNTorsoImpl>(obs_channels, 64, 0);
    networks::StrategoNet champion(torso_champion, action_channels * H * W, H * W);

    // Initial sync
    {
        torch::NoGradGuard no_grad;
        auto params = challenger->parameters();
        auto champ_params = champion->parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            champ_params[i].copy_(params[i]);
        }
    }

    challenger->to(torch::kCPU);
    champion->to(torch::kCPU);

    PPOAgent agent(challenger, 1e-4, 0.99, 1, 0.2); 
    RolloutBuffer<torch::Tensor, int> buffer;

    std::cout << "Starting Self-Play Training (Challenger vs Champion) for " << num_episodes << " episodes..." << std::endl;

    float total_reward = 0;
    int games_count = 0;
    int eval_freq = 500;
    float win_threshold = 0.55f;

    for (int i = 1; i <= num_episodes; ++i) {
        torch::Tensor obs = env->reset();
        bool done = false;
        std::vector<stratego::Player> move_owners;

        while (!done) {
            torch::Tensor mask = env->get_action_mask();
            if (mask.sum().item<float>() == 0) {
                float final_reward = 1.0f;
                stratego::Player winner = (env->get_current_player() == stratego::Player::Red) ? 
                                           stratego::Player::Blue : stratego::Player::Red;
                
                if (!move_owners.empty()) {
                    int start_idx = (int)buffer.size() - (int)move_owners.size();
                    for (int j = 0; j < (int)move_owners.size(); ++j) {
                        float r = (move_owners[j] == winner) ? final_reward : -final_reward;
                        buffer.rewards[start_idx + j] = r;
                        total_reward += r;
                    }
                    buffer.is_terminals.back() = true;
                }
                done = true;
                break;
            }

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

                int start_idx = (int)buffer.size() - (int)move_owners.size();
                for (int j = 0; j < (int)move_owners.size(); ++j) {
                    float r = (move_owners[j] == last_mover) ? final_reward : -final_reward;
                    buffer.rewards[start_idx + j] = r;
                    total_reward += r;
                }
                buffer.is_terminals.back() = true;
            }
        }
        games_count++;

        agent.update_weights(buffer);
        buffer.clear();

        if (i % eval_freq == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            
            std::cout << "\n--- Episode " << i << ": Evaluating Challenger vs Champion ---" << std::endl;
            auto res = evaluate_vs_champion(challenger, champion, config, setup_type, 100);
            std::cout << "Challenger Win Rate: " << res.win_rate * 100 << "% | Draw Rate: " << res.draw_rate * 100 << "%" << std::endl;
            
            if (res.win_rate >= win_threshold) {
                std::cout << ">>> CHALLENGER IS THE NEW CHAMPION! Updating champion... <<<" << std::endl;
                torch::NoGradGuard no_grad;
                auto params = challenger->parameters();
                auto champ_params = champion->parameters();
                for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
                    champ_params[p_idx].copy_(params[p_idx]);
                }
                torch::save(challenger, model_path);
            } else {
                std::cout << "Challenger failed to dethrone the Champion. Continuing training..." << std::endl;
            }
            std::cout << "Time: " << duration << "s" << std::endl;
            std::cout << "---------------------------------------------------------" << std::endl;
            
            total_reward = 0;
            games_count = 0;
        }
    }

    torch::save(challenger, model_path);
    std::cout << "Training complete. Final model saved to " << model_path << std::endl;
    return 0;
}
