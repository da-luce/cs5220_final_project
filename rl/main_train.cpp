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
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>
#include <memory>
#include <algorithm>
#include <chrono>
#include <string>
#include <random>

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

    challenger->train();
    return {(float)challenger_wins / num_games, (float)draws / num_games};
}

EvalResult evaluate_vs_random(
    networks::StrategoNet challenger,
    const stratego::BoardConfig& config,
    stratego::state::SetupType setup_type,
    int num_games = 50
) {
    challenger->eval();
    torch::NoGradGuard no_grad;

    NeuralPolicy net_policy(challenger, config);
    RandomPolicy rand_policy;

    int challenger_wins = 0;
    int random_wins = 0;
    int draws = 0;

    for (int g = 0; g < num_games; ++g) {
        stratego::GameState state = stratego::state::initialize(config, setup_type, 60);
        bool challenger_is_red = (g % 2 == 0);

        while (true) {
            stratego::Player current = state.current_turn;
            bool is_challenger_turn = (current == stratego::Player::Red && challenger_is_red) ||
                                      (current == stratego::Player::Blue && !challenger_is_red);

            if (stratego::Engine::get_all_legal_moves(state, current).empty()) {
                if (is_challenger_turn) random_wins++;
                else challenger_wins++;
                break;
            }

            stratego::GameState view = stratego::state::get_masked_view(state, current);
            stratego::Policy* active = is_challenger_turn ? (stratego::Policy*)&net_policy
                                                          : (stratego::Policy*)&rand_policy;
            stratego::Move move = active->get_move(view);
            stratego::CombatResult outcome = stratego::Engine::execute_move(state, move);

            if (outcome == stratego::CombatResult::FlagCaptured) {
                if (is_challenger_turn) challenger_wins++;
                else random_wins++;
                break;
            }
            if (outcome == stratego::CombatResult::InvalidMove) {
                if (is_challenger_turn) random_wins++;
                else challenger_wins++;
                break;
            }
            if (state.move_count >= state.max_moves) {
                draws++;
                break;
            }
        }
    }

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

    // Linear scaling for SGD is common approach: https://arxiv.org/pdf/1706.02677.pdf
    // TODO: add warmup for stability at the start of training, especially for larger batch sizes
    int effective_batch_size = batch_size * world_size;
    double lr = (effective_batch_size / 256.0) * 1e-4;
    const double gamma    = 0.99;
    const int    k_epochs = 1;
    const double eps_clip = 0.2;

    PPOAgent agent(challenger, lr, gamma, k_epochs, eps_clip);

    RolloutBuffer<torch::Tensor, int> buffer;

    int local_episodes = num_episodes / world_size;
    int sync_freq = batch_size;
    const int eval_every_batches = 2;  // ~512 games per eval at sync_freq=256

    std::ofstream metrics_log;
    std::string metrics_path;
    if (rank == 0) {
        std::cout << "\n[Build Info]\n"
                  << "Ranks: " << world_size
                  << " | Per-Rank Batch: " << batch_size
                  << " | Effective Batch: " << effective_batch_size << "\n"
                  << "Learning Rate: " << std::scientific << std::setprecision(2) << lr
                  << " (Linear Scaling Applied)\n"
                  << std::defaultfloat << std::setprecision(6)
                  << "Optimizer: Adam | Clip: " << eps_clip << " | Gamma: " << gamma
                  << std::endl;

        std::time_t t0 = std::time(nullptr);
        std::tm tm_buf;
        localtime_r(&t0, &tm_buf);
        char stamp[32], iso[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf);
        std::strftime(iso,   sizeof(iso),   "%Y-%m-%dT%H:%M:%S", &tm_buf);

        metrics_path = std::string(PROJECT_ROOT_DIR) + "/logs/metrics_"
                     + variant_str + "_" + setup_str + "_" + stamp + ".jsonl";
        metrics_log.open(metrics_path);
        if (metrics_log.is_open()) {
            metrics_log << std::setprecision(8);
            metrics_log << "{\"meta\":true"
                        << ",\"started\":\"" << iso << "\""
                        << ",\"num_episodes\":" << num_episodes
                        << ",\"world_size\":" << world_size
                        << ",\"batch_size\":" << batch_size
                        << ",\"sync_freq\":" << sync_freq
                        << ",\"eval_every_batches\":" << eval_every_batches
                        << ",\"learning_rate\":" << lr
                        << ",\"gamma\":" << gamma
                        << ",\"eps_clip\":" << eps_clip
                        << ",\"k_epochs\":" << k_epochs
                        << "}\n";
            metrics_log.flush();
            std::cout << "Metrics log: " << metrics_path << std::endl;
        } else {
            std::cerr << "Warning: failed to open metrics log at " << metrics_path << std::endl;
        }

        std::cout << "\nStarting Self-Play Training (Challenger vs Champion) for "
                  << local_episodes << " episodes per rank (" << num_episodes
                  << " total across " << world_size << " ranks), syncing every "
                  << sync_freq << " episodes..." << std::endl;
    }

    float total_reward = 0;
    int games_count = 0;
    int batch_idx = 0;
    float win_threshold = 0.55f;

    auto json_num = [](float v) -> std::string {
        if (std::isnan(v) || std::isinf(v)) return "null";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        return buf;
    };

    // Per-env PPO tracking
    std::vector<std::vector<stratego::Player>> move_owners(batch_size);
    std::vector<RolloutBuffer<torch::Tensor, int>> env_bufs(batch_size);

    // Pre-allocated inference buffers (filled with the active subset each step)
    auto buf_opts = torch::TensorOptions().dtype(torch::kFloat32)
                        .pinned_memory(torch::cuda::is_available());
    torch::Tensor inf_obs  = torch::zeros({batch_size, obs_channels, H, W}, buf_opts);
    torch::Tensor inf_mask = torch::zeros({batch_size, action_dim_size},     buf_opts);
    torch::Tensor all_actions = torch::zeros({batch_size}, torch::kInt64);

    int last_batch = 0;

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

    while (true) {
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

        // End-of-batch: PPO update, sync, optional eval, log, termination check.
        // All "per-batch" work happens together, so eval/log can use this batch's
        // fresh stats without any pending-state bookkeeping.
        if (games_count - last_batch < sync_freq) continue;
        last_batch = games_count;
        batch_idx++;

        if (buffer.size() > 0) {
            agent.update_weights(buffer);
            buffer.clear();
        }
        sync_weights(challenger, ctx);

        bool did_eval = false;
        bool champion_replaced = false;
        EvalResult res{};
        EvalResult rand_res{};
        if (rank == 0 && batch_idx % eval_every_batches == 0) {
            res = evaluate_vs_champion(challenger, champion, config, setup_type, device, 100);
            rand_res = evaluate_vs_random(challenger, config, setup_type, 100);
            did_eval = true;

            if (res.win_rate >= win_threshold) {
                torch::NoGradGuard no_grad;
                auto params = challenger->parameters();
                auto champ_params = champion->parameters();
                for (size_t p_idx = 0; p_idx < params.size(); ++p_idx)
                    champ_params[p_idx].copy_(params[p_idx]);
                torch::save(challenger, model_path);
                champion_replaced = true;
            }
        }

        if (rank == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            auto stats = agent.last_stats();

            std::cout << "[batch " << batch_idx << "] ep=" << games_count
                      << " t=" << duration << "ms"
                      << std::fixed << std::setprecision(4)
                      << " loss=" << stats.policy_loss
                      << " ent=" << stats.entropy
                      << " kl=" << stats.kl_divergence
                      << std::defaultfloat << std::setprecision(6);
            if (did_eval) {
                std::cout << " | win=" << (int)(res.win_rate * 100)
                          << "% draw=" << (int)(res.draw_rate * 100) << "%"
                          << " vs_rand=" << (int)(rand_res.win_rate * 100)
                          << "% rand_draw=" << (int)(rand_res.draw_rate * 100) << "%"
                          << " champ=" << (champion_replaced ? "REPLACED" : "kept");
            }
            std::cout << std::endl;

            if (metrics_log.is_open()) {
                metrics_log << "{\"batch\":" << batch_idx
                            << ",\"episode\":" << games_count
                            << ",\"time_ms\":" << duration
                            << ",\"policy_loss\":" << json_num(stats.policy_loss)
                            << ",\"value_loss\":" << json_num(stats.value_loss)
                            << ",\"entropy\":" << json_num(stats.entropy)
                            << ",\"kl\":" << json_num(stats.kl_divergence)
                            << ",\"win_rate\":"  << (did_eval ? json_num(res.win_rate)  : "null")
                            << ",\"draw_rate\":" << (did_eval ? json_num(res.draw_rate) : "null")
                            << ",\"random_win_rate\":"  << (did_eval ? json_num(rand_res.win_rate)  : "null")
                            << ",\"random_draw_rate\":" << (did_eval ? json_num(rand_res.draw_rate) : "null")
                            << ",\"champion_replaced\":" << (champion_replaced ? "true" : "false")
                            << "}\n";
                metrics_log.flush();
            }
        }

        // Collective termination — paired one-to-one with sync_weights above.
        if (all_ranks_done(games_count >= local_episodes, ctx)) break;
    }

    if (buffer.size() > 0) {
        agent.update_weights(buffer);
        buffer.clear();
    }

    if (rank == 0) {
        torch::save(challenger, model_path);
        auto end_time  = std::chrono::high_resolution_clock::now();
        auto total_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Training complete. Final model saved to " << model_path << std::endl;
        std::cout << "Total training time: " << total_ms << "ms" << std::endl;

        if (metrics_log.is_open()) {
            metrics_log << "{\"summary\":true"
                        << ",\"total_time_ms\":" << total_ms
                        << ",\"completed_episodes\":" << games_count
                        << "}\n";
            metrics_log.flush();
            metrics_log.close();
        }
    }

    return 0;
}
