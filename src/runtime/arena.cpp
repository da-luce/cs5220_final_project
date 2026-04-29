#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../rl/policy_neural.h"
#include "game_runner.cpp"
#include "policy_random.cpp"
#include "policy_ucc.h"

#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace stratego;

std::unique_ptr<Policy> create_agent(const std::string& spec, Player player, const BoardConfig& config, GameState& state) {
    if (spec == "random") {
        return std::make_unique<Random>();
    } else if (spec.find("neural:") == 0) {
        std::string path = spec.substr(7);
        return std::make_unique<NeuralPolicy>(path, config);
    } else if (spec.find("ucc:") == 0) {
        std::string path = spec.substr(4);
        return std::make_unique<PolicyUCC>(path, player, state);
    }
    throw std::invalid_argument("Unknown agent spec: " + spec);
}

int main(int argc, char** argv) {
    std::string red_spec = "random";
    std::string blue_spec = "random";
    int num_games = 1;
    int num_threads = 0; // 0 = let OpenMP decide

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--red" && i + 1 < argc) {
            red_spec = argv[++i];
        } else if (arg == "--blue" && i + 1 < argc) {
            blue_spec = argv[++i];
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [--red <spec>] [--blue <spec>] [--games <N>] [--threads <T>]\n";
            std::cerr << "Specs: random, neural:<path>, ucc:<path>\n";
            return 1;
        }
    }

#ifdef _OPENMP
    if (num_threads > 0)
        omp_set_num_threads(num_threads);
    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    std::cout << "Parallelism: OpenMP, " << actual_threads << " thread(s)\n";
#else
    std::cout << "Parallelism: NONE (compiled without OpenMP — games run serially)\n";
    if (num_threads > 0)
        std::cerr << "Warning: --threads ignored, OpenMP not available\n";
#endif

    std::cout << "Arena configured:\n";
    std::cout << "Red:  " << red_spec << "\n";
    std::cout << "Blue: " << blue_spec << "\n";
    std::cout << "Games: " << num_games << "\n\n";

    int red_wins = 0;
    int blue_wins = 0;
    int draws = 0;

    BoardConfig config = get_config_for_game_type(GameType::Classic);
    std::atomic<int> games_done{0};

    auto t_start = std::chrono::steady_clock::now();

    #pragma omp parallel for reduction(+:red_wins,blue_wins,draws) schedule(dynamic)
    for (int i = 0; i < num_games; ++i) {
        try {
            GameState state = state::initialize(config, state::SetupType::Random, 2000);

            auto red_agent = create_agent(red_spec, Player::Red, config, state);
            auto blue_agent = create_agent(blue_spec, Player::Blue, config, state);

            GameRunner runner(state, std::move(red_agent), std::move(blue_agent));

            CombatResult cr = CombatOutcome::MovedToEmpty;
            while (cr == CombatOutcome::MovedToEmpty || cr == CombatOutcome::AttackerWins ||
                   cr == CombatOutcome::DefenderWins || cr == CombatOutcome::BothDestroyed) {
                cr = runner.step();
            }

            int done = ++games_done;

            if (cr == CombatOutcome::FlagCaptured) {
                if (runner.get_state().current_turn == Player::Red) {
                    blue_wins++;
                } else {
                    red_wins++;
                }
                #pragma omp critical
                std::cout << "[" << done << "/" << num_games << "] Flag Captured\n" << std::flush;
            } else if (cr == CombatOutcome::Draw) {
                draws++;
                #pragma omp critical
                std::cout << "[" << done << "/" << num_games << "] Draw\n" << std::flush;
            } else if (cr == CombatOutcome::NoLegalMoves) {
                Player loser = runner.get_last_attempted_player();
                const char* who = (loser == Player::Red) ? "RED" : "BLUE";
                if (loser == Player::Red) blue_wins++;
                else red_wins++;
                #pragma omp critical
                std::cout << "[" << done << "/" << num_games << "] No Legal Moves — " << who << " loses\n" << std::flush;
            } else if (cr == CombatOutcome::InvalidMove) {
                Move last = runner.get_last_attempted_move();
                Player loser = runner.get_last_attempted_player();
                const char* who = (loser == Player::Red) ? "RED" : "BLUE";
                if (loser == Player::Red) blue_wins++;
                else red_wins++;
                #pragma omp critical
                std::cout << "[" << done << "/" << num_games << "] Invalid Move by " << who
                          << " (" << last.start_x << "," << last.start_y
                          << " -> " << last.end_x << "," << last.end_y << ")\n" << std::flush;
            } else {
                draws++;
                #pragma omp critical
                std::cout << "[" << done << "/" << num_games << "] Other ("
                          << static_cast<int>(cr.outcome) << ")\n" << std::flush;
            }
        } catch (const std::exception& e) {
            ++games_done;
            draws++;
            #pragma omp critical
            std::cout << "Error in game " << games_done.load() << ": " << e.what() << "\n" << std::flush;
        } catch (...) {
            ++games_done;
            draws++;
            #pragma omp critical
            std::cout << "Unknown error in game " << games_done.load() << "\n" << std::flush;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== Arena Results ===\n";
    std::cout << "Total Games: " << num_games << "\n";
    std::cout << "Red Wins:    " << red_wins << "\n";
    std::cout << "Blue Wins:   " << blue_wins << "\n";
    std::cout << "Draws:       " << draws << "\n";
    std::cout << "Time:        " << elapsed << "s ("
              << (num_games / elapsed) << " games/s)\n";

    return 0;
}
