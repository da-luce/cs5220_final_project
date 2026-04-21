#include "../environment/stratego_env.cpp"
#include "stratego_tiny_agent.cpp"
#include "rollout_buffer.cpp"
#include "ppo.cpp"
#include <iostream>
#include <string>
#include <vector>

void evaluate_vs_champion(StrategoEnvironment& env, StrategoTinyAgent& challenger, StrategoTinyAgent& champion, int num_games, float& out_win_rate, float& out_draw_rate) {
    challenger.set_eval_mode(true);
    champion.set_eval_mode(true);

    int challenger_wins = 0;
    int draws = 0;

    for (int game = 0; game < num_games; ++game) {
        auto obs = env.reset();
        bool done = false;
        bool challenger_is_p0 = (game % 2 == 0);
        
        while (!done) {
            auto current_player = env.get_board().get_current_turn();
            bool is_p0 = (current_player == stratego::Player::Red);
            
            StrategoTinyAgent* active_net = nullptr;
            if ((is_p0 && challenger_is_p0) || (!is_p0 && !challenger_is_p0)) {
                active_net = &challenger;
            } else {
                active_net = &champion;
            }

            auto out = active_net->act(obs);
            auto next = env.step(out.action);
            
            obs = next.observation;
            done = next.terminated || next.truncated;

            if (done) {
                if (next.truncated) {
                    draws++;
                } else if (next.reward > 0.5f) {
                    if (active_net == &challenger) challenger_wins++;
                } else if (next.reward < -0.5f) {
                    if (active_net != &challenger) challenger_wins++;
                } else {
                    draws++;
                }
            }
        }
    }
    
    challenger.set_eval_mode(false);
    champion.set_eval_mode(false);

    out_win_rate = static_cast<float>(challenger_wins) / num_games;
    out_draw_rate = static_cast<float>(draws) / num_games;
}

int main(int argc, char* argv[]) {
    std::string output_file = "stratego_best_model.pth";
    int max_iterations = 20000;
    int eval_freq = 10;
    int eval_games = 100;
    float win_threshold = 0.55f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--eval-freq" && i + 1 < argc) {
            eval_freq = std::stoi(argv[++i]);
        }
    }

    std::cout << "Starting Batched Self-Play Training...\n";
    std::cout << "Output File: " << output_file << "\n";
    std::cout << "Max Iterations: " << max_iterations << "\n\n";

    stratego::GameConfig config = stratego::get_config_for_game_type(stratego::GameType::Tiny);
    stratego::Board board(config);
    board.initialize_game(stratego::SetupType::Random);
    StrategoEnvironment env(board);
    StrategoTinyAgent challenger(env);
    StrategoTinyAgent champion(env);

    champion.copy_weights_from(challenger);
    champion.save_model("stratego_random_weights.pth");

    RolloutBuffer<StrategoObs, StrategoAction> buffer;
    
    // PPO Setup
    int rollout_length = 2048; // Common standard sequence size for batched rollouts
    int current_epoch = 0;

    // Callback executes at the end of each rollout step inside `train_ppo`
    auto eval_cb = [&](int it) {
        current_epoch++;
        
        if (current_epoch % 10 == 0 || current_epoch == 1) {
            std::cout << "Completed rollout " << current_epoch << " / " << max_iterations << " (" << it << " timesteps)\n";
        }

        if (current_epoch % eval_freq == 0) {
            std::cout << "--- Iteration " << it << ": Evaluating Challenger vs Champion ---\n";
            float win_rate = 0.0f, draw_rate = 0.0f;
            evaluate_vs_champion(env, challenger, champion, eval_games, win_rate, draw_rate);
            std::cout << "Challenger Win Rate: " << (win_rate * 100.0f) << "% | Draw Rate: " << (draw_rate * 100.0f) << "%\n";
            if (win_rate >= win_threshold) {
                std::cout << ">>> CHALLENGER IS THE NEW CHAMPION! Saving model... <<<\n";
                champion.copy_weights_from(challenger);
                champion.save_model(output_file);
            }
            std::cout << "--------------------------------------------------\n";
        }
    };

    train_ppo(env, challenger, buffer, max_iterations * rollout_length, rollout_length, eval_cb);

    return 0;
}