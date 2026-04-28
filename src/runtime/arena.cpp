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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--red" && i + 1 < argc) {
            red_spec = argv[++i];
        } else if (arg == "--blue" && i + 1 < argc) {
            blue_spec = argv[++i];
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [--red <spec>] [--blue <spec>] [--games <N>]\n";
            std::cerr << "Specs: random, neural:<path>, ucc:<path>\n";
            return 1;
        }
    }

    std::cout << "Arena configured:\n";
    std::cout << "Red:  " << red_spec << "\n";
    std::cout << "Blue: " << blue_spec << "\n";
    std::cout << "Games: " << num_games << "\n\n";

    int red_wins = 0;
    int blue_wins = 0;
    int draws = 0;

    BoardConfig config = get_config_for_game_type(GameType::Classic);

    for (int i = 0; i < num_games; ++i) {
        std::cout << "\rSimulating game " << (i + 1) << " of " << num_games << "..." << std::flush;
        
        try {
            GameState state = state::initialize(config, state::SetupType::Random, 2000);

            auto red_agent = create_agent(red_spec, Player::Red, config, state);
            auto blue_agent = create_agent(blue_spec, Player::Blue, config, state);

            GameRunner runner(state, std::move(red_agent), std::move(blue_agent));

            CombatResult cr = CombatResult::MovedToEmpty;
            while (cr == CombatResult::MovedToEmpty || cr == CombatResult::AttackerWins || 
                   cr == CombatResult::DefenderWins || cr == CombatResult::BothDestroyed) {
                cr = runner.step();
            }

            if (cr == CombatResult::FlagCaptured) {
                std::cout << "\nResult: Flag Captured!\n";
                // The player whose turn it is when the flag is captured is the
                // loser, since they failed to prevent the capture on their turn.
                if (runner.get_state().current_turn == Player::Red) {
                    blue_wins++;
                } else {
                    red_wins++;
                }
            } else if (cr == CombatResult::Draw) {
                std::cout << "\nResult: Draw!\n";
                draws++;
            } else if (cr == CombatResult::NoLegalMoves) {
                Player loser = runner.get_last_attempted_player();
                const char* who = (loser == Player::Red) ? "RED" : "BLUE";
                std::cout << "\nResult: " << who << " has no legal moves — loses.\n";
                if (loser == Player::Red) {
                    blue_wins++;
                } else {
                    red_wins++;
                }
            } else if (cr == CombatResult::InvalidMove) {
                // Invalid moves are NOT added to move_history; pull the
                // attempted move directly from the runner.
                Move last = runner.get_last_attempted_move();
                Player loser = runner.get_last_attempted_player();
                const char* who = (loser == Player::Red) ? "RED" : "BLUE";
                std::cout << "\nResult: Invalid Move by " << who << "! ("
                          << last.start_x << "," << last.start_y << " -> "
                          << last.end_x << "," << last.end_y << ")\n";
                if (loser == Player::Red) {
                    blue_wins++;
                } else {
                    red_wins++;
                }
            } else {
                std::cout << "\nResult: Other (" << static_cast<int>(cr) << ")\n";
                draws++;
            }
        } catch (const std::exception& e) {
            std::cout << "\nError in game " << (i + 1) << ": " << e.what() << "\n";
            draws++; // Count as draw or handled differently
        } catch (...) {
            std::cout << "\nUnknown error in game " << (i + 1) << "\n";
            draws++;
        }
    }

    std::cout << "\n=== Arena Results ===\n";
    std::cout << "Total Games: " << num_games << "\n";
    std::cout << "Red Wins:    " << red_wins << "\n";
    std::cout << "Blue Wins:   " << blue_wins << "\n";
    std::cout << "Draws:       " << draws << "\n";

    return 0;
}