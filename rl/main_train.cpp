#include "rl/environment/stratego_env.h"
#include "rl/environment/batched_stratego_env.h"
#include "rl/agent/ppo_agent.h"
#include "rl/agent/random_policy.h"
#include "rl/policy_neural.h"
#include "networks/model.h"
#include "networks/torsos/cnn.h"
#include "rl/encoding/board.h"
#include "rl/distributed.h"
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
    torch::Device device,
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

        while (!done) {
            torch::Tensor mask = env.get_action_mask();
            stratego::Player current_p = env.get_current_player();
            bool is_challenger_turn = (current_p == stratego::Player::Red && challenger_is_red) ||
                                      (current_p == stratego::Player::Blue && !challenger_is_red);

            if (mask.sum().item<float>() == 0) {
                if (is_challenger_turn) champion_wins++;
                else challenger_wins++;
                done = true;
                break;
            }

            auto& active_net = is_challenger_turn ? challenger : champion;
            auto [logits, value] = active_net->forward(obs.to(device).unsqueeze(0));

            torch::Tensor masked_logits = logits.view({1, -1}).clone();
            masked_logits.masked_fill_(mask.to(device).unsqueeze(0) == 0, -1e9);

            int action = torch::argmax(masked_logits, 1).item<int>();
            auto step_res = env.step(action);
            obs = step_res.observation;
            done = step_res.terminated;

            if (done) {
                if (step_res.reward == 1.0f) {
                    if (is_challenger_turn) challenger_wins++;
                    else champion_wins++;
                } else if (step_res.reward == -1.0f) {
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
    std::cout.setf(std::ios::unitbuf);
    torch::set_num_threads(1);
    auto start_time = std::chrono::high_resolution_clock::now();

    DistributedContext ctx(argc, argv);
    const int rank       = ctx.rank;
    const int world_size = ctx.world_size;
    torch::Device device = ctx.device();

    std::string variant_str = "tiny";
    std::string setup_str = "random";
    int num_episodes = 20000;
    int batch_size = 256;
    if (argc >= 2) variant_str = argv[1];
    if (argc >= 3) setup_str = argv[2];
    if (argc >= 4) num_episodes = std::stoi(argv[3]);
    if (argc >= 5) batch_size = std::stoi(argv[4]);

    stratego::GameType game_type;
    if (variant_str == "classic")       game_type = stratego::GameType::Classic;
    else if (variant_str == "quick")    game_type = stratego::GameType::Quick;
    else if (variant_str == "barrage")  game_type = stratego::GameType::Barrage;
    else                                game_type = stratego::GameType::Tiny;

    stratego::state::SetupType setup_type;
    if (setup_str == "default")             setup_type = stratego::state::SetupType::Default;
    else if (setup_str == "probabilistic")  setup_type = stratego::state::SetupType::Probabilistic;
    else                                    setup_type = stratego::state::SetupType::Random;

    stratego::BoardConfig config = stratego::get_config_for_game_type(game_type);

    std::string model_name = variant_str + "_" + setup_str + ".pt";
    std::string model_path = std::string(PROJECT_ROOT_DIR) + "/models/" + model_name;

    BatchedStrategoEnv batched_env(config, setup_type, batch_size, 60);

    int obs_channels    = get_encoding_channels(config);
    int action_channels = batched_env.get_action_channels();
    int H = config.height;
    int W = config.width;
    int action_dim_size = action_channels * H * W;

    if (rank == 0) {
        std::cout << "Config: Obs Channels=" << obs_channels << " Action Channels=" << action_channels << " Dim=" << H << "x" << W << std::endl;
        std::cout << "Target Path: " << model_path << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }

    auto torso_challenger = std::make_shared<networks::torsos::CNNTorsoImpl>(obs_channels, 64, 0);
    networks::StrategoNet challenger(torso_challenger, action_channels * H * W, H * W);

    auto torso_champion = std::make_shared<networks::torsos::CNNTorsoImpl>(obs_channels, 64, 0);
    networks::StrategoNet champion(torso_champion, action_channels * H * W, H * W);

    {
        torch::NoGradGuard no_grad;
        auto params = challenger->parameters();
        auto champ_params = champion->parameters();
        for (size_t i = 0; i < params.size(); ++i)
            champ_params[i].copy_(params[i]);
    }

    challenger->to(device);
    champion->to(device);

    PPOAgent agent(challenger, 1e-4, 0.99, 1, 0.2);
    RolloutBuffer<torch::Tensor, int> buffer;

    int local_episodes = num_episodes / world_size;
    int sync_freq = batch_size;

    if (rank == 0) {
        std::cout << "Starting Self-Play Training (Challenger vs Champion) for "
                  << local_episodes << " episodes per rank (" << num_episodes
                  << " total across " << world_size << " ranks), syncing every "
                  << sync_freq << " episodes..." << std::endl;
    }

    float total_reward = 0;
    int games_count = 0;
    int eval_freq = 500;
    float win_threshold = 0.55f;

    // Per-env PPO tracking
    std::vector<std::vector<stratego::Player>> move_owners(batch_size);
    std::vector<RolloutBuffer<torch::Tensor, int>> env_bufs(batch_size);

    // Pre-allocated inference buffers (filled with the active subset each step)
    auto buf_opts = torch::TensorOptions().dtype(torch::kFloat32)
                        .pinned_memory(torch::cuda::is_available());
    torch::Tensor inf_obs  = torch::zeros({batch_size, obs_channels, H, W}, buf_opts);
    torch::Tensor inf_mask = torch::zeros({batch_size, action_dim_size},     buf_opts);
    torch::Tensor all_actions = torch::zeros({batch_size}, torch::kInt64);

    int last_sync = 0, last_eval = 0, last_update = 0;

    auto merge_env_buf = [&](int e) {
        for (size_t j = 0; j < env_bufs[e].size(); ++j) {
            buffer.observations.push_back(env_bufs[e].observations[j]);
            buffer.actions.push_back(env_bufs[e].actions[j]);
            buffer.log_probs.push_back(env_bufs[e].log_probs[j]);
            buffer.values.push_back(env_bufs[e].values[j]);
            buffer.masks.push_back(env_bufs[e].masks[j]);
            buffer.rewards.push_back(env_bufs[e].rewards[j]);
            buffer.is_terminals.push_back(env_bufs[e].is_terminals[j]);
        }
        env_bufs[e].clear();
        move_owners[e].clear();
    };

    // cur_obs/cur_masks alias the batched env's internal pinned buffers and stay
    // in sync after every step() call (obs_buf/mask_buf are updated in-place).
    auto [cur_obs, cur_masks] = batched_env.reset_all();

    while (games_count < local_episodes) {
        auto current_players = batched_env.get_current_players();

        std::vector<int> active;
        active.reserve(batch_size);
        std::vector<bool> is_trap(batch_size, false);
        all_actions.zero_();

        for (int e = 0; e < batch_size; ++e) {
            if (cur_masks[e].sum().item<float>() == 0) {
                // Trapped: current player has no legal moves — opponent wins
                is_trap[e] = true;
                stratego::Player winner = (current_players[e] == stratego::Player::Red)
                                          ? stratego::Player::Blue : stratego::Player::Red;
                for (int j = 0; j < (int)move_owners[e].size(); ++j) {
                    float r = (move_owners[e][j] == winner) ? 1.0f : -1.0f;
                    env_bufs[e].rewards[j] = r;
                    total_reward += r;
                }
                if (!env_bufs[e].is_terminals.empty())
                    env_bufs[e].is_terminals.back() = true;
                merge_env_buf(e);
                games_count++;
                // all_actions[e] stays 0 — the env will treat it as an invalid move,
                // terminate, and immediately auto-reset inside step().
            } else {
                int ai = (int)active.size();
                inf_obs[ai].copy_(cur_obs[e]);
                inf_mask[ai].copy_(cur_masks[e]);
                active.push_back(e);
            }
        }

        if (!active.empty()) {
            int n = (int)active.size();
            auto outputs = agent.act_batch(
                inf_obs.slice(0, 0, n).to(device, /*non_blocking=*/true),
                inf_mask.slice(0, 0, n).to(device, /*non_blocking=*/true)
            );

            // Record pre-step data (clone because obs_buf is updated in-place by step)
            for (int ai = 0; ai < n; ++ai) {
                int e = active[ai];
                move_owners[e].push_back(current_players[e]);
                env_bufs[e].observations.push_back(cur_obs[e].clone());
                env_bufs[e].actions.push_back(outputs.actions[ai]);
                env_bufs[e].log_probs.push_back(outputs.log_probs[ai]);
                env_bufs[e].values.push_back(outputs.values[ai]);
                env_bufs[e].masks.push_back(cur_masks[e].clone());
                env_bufs[e].rewards.push_back(0.0f);
                env_bufs[e].is_terminals.push_back(false);
                all_actions[e] = (int64_t)outputs.actions[ai];
            }
        }

        // Step all envs; terminated ones immediately auto-reset inside step()
        auto result = batched_env.step(all_actions);

        // Handle terminals for active (non-trapped) envs
        for (int ai = 0; ai < (int)active.size(); ++ai) {
            int e = active[ai];
            if (result.dones[e]) {
                float fr = result.rewards[e];
                stratego::Player last_mover = move_owners[e].back();
                for (int j = 0; j < (int)move_owners[e].size(); ++j) {
                    float r = (move_owners[e][j] == last_mover) ? fr : -fr;
                    env_bufs[e].rewards[j] = r;
                    total_reward += r;
                }
                env_bufs[e].is_terminals.back() = true;
                merge_env_buf(e);
                games_count++;
            }
        }

        // cur_obs/cur_masks alias obs_buf/mask_buf which were updated in-place by step().
        // Re-assign to pick up the returned tensors explicitly (no-op in practice).
        cur_obs   = result.observations;
        cur_masks = result.masks;

        // PPO update every batch_size completed games
        if (games_count - last_update >= batch_size && buffer.size() > 0) {
            agent.update_weights(buffer);
            buffer.clear();
            last_update = games_count;
        }

        if (games_count - last_sync >= sync_freq) {
            last_sync = games_count;
            sync_weights(challenger, ctx);
        }

        if (rank == 0 && games_count - last_eval >= eval_freq && games_count > 0) {
            last_eval = games_count;
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            std::cout << "\n--- Episode " << games_count << ": Evaluating Challenger vs Champion ---" << std::endl;
            auto res = evaluate_vs_champion(challenger, champion, config, setup_type, device, 100);
            std::cout << "Challenger Win Rate: " << res.win_rate * 100 << "% | Draw Rate: " << res.draw_rate * 100 << "%" << std::endl;

            if (res.win_rate >= win_threshold) {
                std::cout << ">>> CHALLENGER IS THE NEW CHAMPION! Updating champion... <<<" << std::endl;
                torch::NoGradGuard no_grad;
                auto params = challenger->parameters();
                auto champ_params = champion->parameters();
                for (size_t p_idx = 0; p_idx < params.size(); ++p_idx)
                    champ_params[p_idx].copy_(params[p_idx]);
                torch::save(challenger, model_path);
            } else {
                std::cout << "Challenger failed to dethrone the Champion. Continuing training..." << std::endl;
            }
            std::cout << "Time: " << duration << "s" << std::endl;
            std::cout << "---------------------------------------------------------" << std::endl;
        }
    }

    if (buffer.size() > 0) {
        agent.update_weights(buffer);
        buffer.clear();
    }

    if (rank == 0) {
        torch::save(challenger, model_path);
        std::cout << "Training complete. Final model saved to " << model_path << std::endl;
    }

    return 0;
}
