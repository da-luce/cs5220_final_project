#include "environment/stratego_tiny.cpp"
#include "rl/stratego_tiny_agent.cpp"
#include "rl/rollout_buffer.cpp"
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
    int eval_freq = 500;
    int eval_games = 100;
    float win_threshold = 0.55f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            max_iterations = std::stoi(argv[++i]);
        }
    }

    std::cout << "Starting Self-Play Training...\n";
    std::cout << "Output File: " << output_file << "\n";
    std::cout << "Max Iterations: " << max_iterations << "\n\n";

    StrategoEnvironment env;
    StrategoTinyAgent challenger(env);
    StrategoTinyAgent champion(env);

    champion.copy_weights_from(challenger);
    champion.save_model("stratego_random_weights.pth");

    RolloutBuffer<StrategoObs, StrategoAction> buffer;

    for (int epoch = 1; epoch <= max_iterations; ++epoch) {
        auto obs = env.reset();
        bool done = false;
        std::vector<stratego::Player> players;

        while (!done) {
            players.push_back(env.get_board().get_current_turn());
            
            auto out = challenger.act(obs);
            auto next = env.step(out.action);

            buffer.add(obs, out.action, next.reward, out.value, out.log_prob, next.terminated || next.truncated, out.mask);
            
            obs = next.observation;
            done = next.terminated || next.truncated;

            if (done) {
                // Standardize episode rewards: +1 for winner's moves, -1 for loser's moves
                size_t step_count = buffer.rewards.size();
                stratego::Player winner = stratego::Player::None;

                if (!next.truncated) {
                    if (next.reward > 0.5f) winner = players.back(); // Last player to act won
                    else if (next.reward < -0.5f) winner = (players.back() == stratego::Player::Red) ? stratego::Player::Blue : stratego::Player::Red;
                }

                for (size_t i = 0; i < step_count; ++i) {
                    if (winner == stratego::Player::None) buffer.rewards[i] = 0.0f;
                    else if (players[i] == winner) buffer.rewards[i] = 1.0f;
                    else buffer.rewards[i] = -1.0f;
                }
                
                buffer.compute_advantages(0.0f, 0.99f, 0.95f);
                challenger.update_weights(buffer);
                buffer.clear();
            }
        }

        if (epoch % eval_freq == 0) {
            std::cout << "--- Epoch " << epoch << ": Evaluating Challenger vs Champion ---\n";
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
    }
    return 0;
}