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

#ifdef USE_NCCL
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#endif

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
            auto [logits, value] = active_net->forward(obs.to(device).unsqueeze(0));

            torch::Tensor masked_logits = logits.view({1, -1}).clone();
            masked_logits.masked_fill_(mask.to(device).unsqueeze(0) == 0, -1e9);

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
    std::cout.setf(std::ios::unitbuf);
    // Stop PyTorch's internal thread pool from fighting our OpenMP threads
    torch::set_num_threads(1);
    auto start_time = std::chrono::high_resolution_clock::now();

    int rank = 0;
    int world_size = 1;

#ifdef USE_NCCL
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    cudaSetDevice(rank);
    if (rank == 0) std::cout << "[NCCL+MPI enabled] MPI world_size=" << world_size << std::endl;
#else
    std::cout << "[CPU-only, no MPI/NCCL] Compiled without USE_NCCL" << std::endl;
#endif

    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, nccl_id, rank);
#endif

    torch::Device device = torch::kCPU;
#ifdef USE_NCCL
    device = torch::Device(torch::kCUDA, rank);
#endif


    std::string variant_str = "tiny";
    std::string setup_str = "random";
    int num_episodes = 20000;
    int batch_size = 256;
    if (argc >= 2) variant_str = argv[1];
    if (argc >= 3) setup_str = argv[2];
    if (argc >= 4) num_episodes = std::stoi(argv[3]);
    if (argc >= 5) batch_size = std::stoi(argv[4]);

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

    std::vector<std::unique_ptr<StrategoEnvironment>> envs(batch_size);
    for (auto& e : envs) e = std::make_unique<StrategoEnvironment>(config, setup_type, 60);

    int obs_channels = get_encoding_channels(config);
    int action_channels = envs[0]->get_action_encoder().get_action_channels();
    int H = config.height;
    int W = config.width;

    if (rank == 0) {
        std::cout << "Config: Obs Channels=" << obs_channels << " Action Channels=" << action_channels << " Dim=" << H << "x" << W << std::endl;
        std::cout << "Target Path: " << model_path << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }

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

    challenger->to(device);
    champion->to(device);

    PPOAgent agent(challenger, 1e-4, 0.99, 1, 0.2); 
    RolloutBuffer<torch::Tensor, int> buffer;

    // Divide work across ranks: each rank does 1/world_size of episodes for 4x wall-clock speedup.
    // Ranks share gradients via AllReduce after each PPO update, so the effective batch size
    // seen per update is batch_size * world_size.
    int local_episodes = num_episodes / world_size;
    int sync_freq = batch_size; // sync once per PPO update, not once per step

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

    // Per-env episode state: each env accumulates its own in-flight episode into
    // env_bufs[e]; completed episodes are merged into the shared buffer for PPO updates.
    std::vector<torch::Tensor> obs_vec(batch_size);
    std::vector<std::vector<stratego::Player>> move_owners_vec(batch_size);
    std::vector<RolloutBuffer<torch::Tensor, int>> env_bufs(batch_size);
    std::vector<bool> needs_reset(batch_size, true);

    // Pre-allocate pinned (page-locked) CPU tensors so the hot loop never allocates.
    // Pinned memory enables async DMA transfers to GPU without CPU involvement.
    int action_dim_size = (int)envs[0]->action_dim();
    auto pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    torch::Tensor obs_buf  = torch::zeros({batch_size, obs_channels, H, W}, pinned);
    torch::Tensor mask_buf = torch::zeros({batch_size, action_dim_size}, pinned);

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
        move_owners_vec[e].clear();
    };

    while (games_count < local_episodes) {
        // Reset finished envs
        // Parallel reset — each env is independent
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < batch_size; ++e) {
            if (needs_reset[e]) {
                obs_vec[e] = envs[e]->reset();
                move_owners_vec[e].clear();
                env_bufs[e].clear();
                needs_reset[e] = false;
            }
        }

        // Parallel mask collection — get_action_mask reads only per-env state
        std::vector<torch::Tensor> all_masks(batch_size);
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < batch_size; ++e) {
            all_masks[e] = envs[e]->get_action_mask();
        }

        // Serial: handle traps; fill pre-allocated pinned buffers for active envs
        std::vector<int> active;
        active.reserve(batch_size);

        for (int e = 0; e < batch_size; ++e) {
            if (all_masks[e].sum().item<float>() == 0) {
                // Trapped: opponent wins
                stratego::Player winner = (envs[e]->get_current_player() == stratego::Player::Red)
                                          ? stratego::Player::Blue : stratego::Player::Red;
                for (int j = 0; j < (int)move_owners_vec[e].size(); ++j) {
                    float r = (move_owners_vec[e][j] == winner) ? 1.0f : -1.0f;
                    env_bufs[e].rewards[j] = r;
                    total_reward += r;
                }
                if (!env_bufs[e].is_terminals.empty())
                    env_bufs[e].is_terminals.back() = true;
                merge_env_buf(e);
                needs_reset[e] = true;
                games_count++;
                continue;
            }
            int ai = (int)active.size();
            obs_buf[ai].copy_(obs_vec[e]);   // write directly into pinned buffer — no alloc
            mask_buf[ai].copy_(all_masks[e]);
            active.push_back(e);
        }

        if (!active.empty()) {
            int n = (int)active.size();
            // Slice the pre-allocated buffers and transfer to GPU (async DMA from pinned mem)
            auto outputs = agent.act_batch(
                obs_buf.slice(0, 0, n).to(device, /*non_blocking=*/true),
                mask_buf.slice(0, 0, n).to(device, /*non_blocking=*/true)
            );

            // Snapshot current players before stepping (step() flips the turn)
            std::vector<stratego::Player> current_players(active.size());
            for (int ai = 0; ai < n; ++ai)
                current_players[ai] = envs[active[ai]]->get_current_player();

            // Parallel env stepping — each env mutates only its own state
            std::vector<StepResult<torch::Tensor>> step_results(active.size());
            #pragma omp parallel for schedule(static)
            for (int ai = 0; ai < n; ++ai)
                step_results[ai] = envs[active[ai]]->step(outputs.actions[ai]);

            // Serial: update per-env buffers and handle episode completion
            for (int ai = 0; ai < n; ++ai) {
                int e = active[ai];
                move_owners_vec[e].push_back(current_players[ai]);

                env_bufs[e].observations.push_back(obs_vec[e]);
                env_bufs[e].actions.push_back(outputs.actions[ai]);
                env_bufs[e].log_probs.push_back(outputs.log_probs[ai]);
                env_bufs[e].values.push_back(outputs.values[ai]);
                env_bufs[e].masks.push_back(all_masks[e]);  // use per-env mask directly
                env_bufs[e].rewards.push_back(0.0f);
                env_bufs[e].is_terminals.push_back(false);

                obs_vec[e] = step_results[ai].observation;

                if (step_results[ai].terminated) {
                    float fr = step_results[ai].reward;
                    stratego::Player last_mover = move_owners_vec[e].back();
                    for (int j = 0; j < (int)move_owners_vec[e].size(); ++j) {
                        float r = (move_owners_vec[e][j] == last_mover) ? fr : -fr;
                        env_bufs[e].rewards[j] = r;
                        total_reward += r;
                    }
                    env_bufs[e].is_terminals.back() = true;
                    merge_env_buf(e);
                    needs_reset[e] = true;
                    games_count++;
                }
            }
        }

        // PPO update every batch_size completed games
        if (games_count - last_update >= batch_size && buffer.size() > 0) {
            agent.update_weights(buffer);
            buffer.clear();
            last_update = games_count;
        }

#ifdef USE_NCCL
        if (games_count - last_sync >= sync_freq) {
            last_sync = games_count;
            torch::NoGradGuard no_grad;
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            for (auto& param : challenger->parameters()) {
                ncclAllReduce(param.data_ptr<float>(), param.data_ptr<float>(),
                              param.numel(), ncclFloat, ncclSum, comm, stream);
            }
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            for (auto& param : challenger->parameters()) {
                param.div_(world_size);
            }
        }
#endif

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
                for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
                    champ_params[p_idx].copy_(params[p_idx]);
                }
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

#ifdef USE_NCCL
    ncclCommDestroy(comm);
    MPI_Finalize();
#endif

    return 0;
}
