#include "stratego.h"
#include "setup_generator.h"
#include "rules_engine.h"
#include "game_serializer.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <random>
#include <iostream>
#include <stdexcept>

namespace stratego {

Board::Board(int w, int h, int max_m) : width(w), height(h), grid(w * h), current_turn(Player::Red), max_moves(max_m), move_count(0) {
    initialize_empty();
}

void Board::initialize_empty() {
    grid.assign(width * height, Piece{PieceType::Empty, Player::None, false});
    current_turn = Player::Red;
    move_count = 0;
    move_history.clear();
    red_chase_hashes.clear();
    blue_chase_hashes.clear();

    // Standard water placement for 10x10 board
    if (width == 10 && height == 10) {
        for (int y = 4; y <= 5; ++y) {
            for (int x = 2; x <= 3; ++x) {
                grid[index(x, y)] = Piece{PieceType::Water, Player::None, true};
            }
            for (int x = 6; x <= 7; ++x) {
                grid[index(x, y)] = Piece{PieceType::Water, Player::None, true};
            }
        }
    } else if (width == 8 && height == 8) {
        for (int y = 3; y <= 4; ++y) {
            grid[index(2, y)] = Piece{PieceType::Water, Player::None, true};
            grid[index(5, y)] = Piece{PieceType::Water, Player::None, true};
        }
    }
}

void Board::initialize_game(GameType type, SetupType setup) {
    if (type == GameType::Classic) {
        width = 10;
        height = 10;
        initialize_empty();

        bool use_probabilistic = (setup == SetupType::Probabilistic) || (setup == SetupType::Default);

        if (use_probabilistic) {
            try {
                // The path is relative to the build directory where the executable runs.
                auto distributions = stratego::setup::load_distributions_from_json("../data/piece_dist.json");
                
                // Generate setups for both players using the new probabilistic generator.
                stratego::setup::generate_probabilistic_setup(*this, Player::Red, distributions);
                stratego::setup::generate_probabilistic_setup(*this, Player::Blue, distributions);
                return;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load probabilistic setup from JSON (" << e.what() << ").\n"
                          << "Falling back to random setup." << std::endl;
            }
        }
        
        // Fallback to the original random setup if the JSON is missing or invalid.
        auto setup_army = [&](Player player, int start_y) {
                std::vector<PieceType> deck = {
                    PieceType::Flag, PieceType::Marshal, PieceType::General, PieceType::Spy
                };
                deck.insert(deck.end(), 6, PieceType::Bomb);
                deck.insert(deck.end(), 8, PieceType::Scout);
                deck.insert(deck.end(), 5, PieceType::Miner);
                deck.insert(deck.end(), 4, PieceType::Sergeant);
                deck.insert(deck.end(), 4, PieceType::Lieutenant);
                deck.insert(deck.end(), 4, PieceType::Captain);
                deck.insert(deck.end(), 3, PieceType::Major);
                deck.insert(deck.end(), 2, PieceType::Colonel);

                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(deck.begin(), deck.end(), g);

                int idx = 0;
                for (int y = start_y; y < start_y + 4; ++y) {
                    for (int x = 0; x < 10; ++x) {
                        place_piece(x, y, deck[idx++], player);
                    }
                }
            };
            setup_army(Player::Blue, 0);
            setup_army(Player::Red, 6);
    } else if (type == GameType::Tiny) {
        width = 4;
        height = 4;
        initialize_empty();

        std::vector<PieceType> deck = {PieceType::Flag, PieceType::Major, PieceType::Captain, PieceType::Lieutenant};
        std::random_device rd;
        std::mt19937 g(rd());
        
        std::shuffle(deck.begin(), deck.end(), g);
        for (int x = 0; x < 4; ++x) place_piece(x, 0, deck[x], Player::Blue);
        
        std::shuffle(deck.begin(), deck.end(), g);
        for (int x = 0; x < 4; ++x) place_piece(x, 3, deck[x], Player::Red);
    } else if (type == GameType::Quick) {
        width = 8;
        height = 8;
        initialize_empty();

        auto setup_quick_army = [&](Player player, int start_y) {
            std::vector<PieceType> deck = {
                PieceType::Flag, PieceType::Spy, PieceType::Marshal, PieceType::General,
                PieceType::Bomb, PieceType::Bomb, PieceType::Miner, PieceType::Miner,
                PieceType::Scout, PieceType::Scout
            };
            // Pad the remaining 6 squares of the 2-row setup area with Empty pieces
            deck.insert(deck.end(), 6, PieceType::Empty);

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(deck.begin(), deck.end(), g);

            int idx = 0;
            for (int y = start_y; y < start_y + 2; ++y) {
                for (int x = 0; x < 8; ++x) {
                    if (deck[idx] != PieceType::Empty) place_piece(x, y, deck[idx], player);
                    idx++;
                }
            }
        };
        
        setup_quick_army(Player::Blue, 0);
        setup_quick_army(Player::Red, 6);
    } else if (type == GameType::Barrage) {
        width = 10;
        height = 10;
        initialize_empty();

        auto setup_barrage_army = [&](Player player, int start_y) {
            std::vector<PieceType> deck = {
                PieceType::Flag, PieceType::Marshal, PieceType::General,
                PieceType::Miner, PieceType::Spy, PieceType::Bomb,
                PieceType::Scout, PieceType::Scout
            };
            // Pad the remaining 32 squares of the 4-row (40 squares) setup area with Empty pieces
            deck.insert(deck.end(), 32, PieceType::Empty);

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(deck.begin(), deck.end(), g);

            int idx = 0;
            for (int y = start_y; y < start_y + 4; ++y) {
                for (int x = 0; x < 10; ++x) {
                    if (deck[idx] != PieceType::Empty) place_piece(x, y, deck[idx], player);
                    idx++;
                }
            }
        };
        setup_barrage_army(Player::Blue, 0);
        setup_barrage_army(Player::Red, 6);
    }
}

bool Board::place_piece(int x, int y, PieceType type, Player owner) {
    if (x < 0 || x >= width || y < 0 || y >= height) return false;
    int idx = index(x, y);
    if (!grid[idx].is_empty()) return false;
    
    grid[idx] = Piece{type, owner, false};
    return true;
}

std::string Board::get_board_state() const {
    std::string state;
    state.reserve(width * height * 2);
    for (const auto& p : grid) {
        state += static_cast<char>(p.type);
        state += static_cast<char>(p.owner);
    }
    return state;
}

bool Board::is_legal_move(const Move& move) const {
    return RulesEngine::is_legal_move(*this, move);
}

std::vector<Move> Board::get_all_legal_moves(Player player) const {
    return RulesEngine::get_all_legal_moves(*this, player);
}

std::vector<Move> Board::get_legal_moves_for_piece(Player player, int x, int y) const {
    return RulesEngine::get_legal_moves_for_piece(*this, player, x, y);
}

bool Board::save_to_file(const std::string& filename) const {
    return GameSerializer::save_to_file(*this, filename);
}

bool Board::load_from_file(const std::string& filename) {
    return GameSerializer::load_from_file(*this, filename);
}

CombatResult Board::execute_move(const Move& move) {
    if (!RulesEngine::is_legal_move(*this, move)) return CombatResult::InvalidMove;

    if (!move_history.empty()) {
        const Move& opp_last = move_history.back();
        int dist = std::abs(move.start_x - opp_last.end_x) + std::abs(move.start_y - opp_last.end_y);
        if (dist > 1) { 
            if (current_turn == Player::Red) blue_chase_hashes.clear();
            else red_chase_hashes.clear();
        }
    }

    int start_idx = index(move.start_x, move.start_y);
    int end_idx = index(move.end_x, move.end_y);

    Piece attacker = grid[start_idx];
    Piece defender = grid[end_idx];

    CombatResult result = CombatResult::MovedToEmpty;

    if (defender.is_empty()) {
        grid[end_idx] = attacker;
        grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
        result = CombatResult::MovedToEmpty;
    } else {
        attacker.revealed = true;
        defender.revealed = true;

        if (defender.type == PieceType::Flag) {
            grid[end_idx] = attacker;
            grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
            result = CombatResult::FlagCaptured;
        } else if (defender.type == PieceType::Bomb) {
            if (attacker.type == PieceType::Miner) {
                grid[end_idx] = attacker;
                grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::AttackerWins;
            } else {
                grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
                grid[end_idx] = defender;
                result = CombatResult::DefenderWins;
            }
        } else if (attacker.type == PieceType::Spy && defender.type == PieceType::Marshal) {
            grid[end_idx] = attacker;
            grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
            result = CombatResult::AttackerWins;
        } else {
            int attacker_val = static_cast<int>(attacker.type);
            int defender_val = static_cast<int>(defender.type);

            if (attacker_val > defender_val) {
                grid[end_idx] = attacker;
                grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::AttackerWins;
            } else if (attacker_val < defender_val) {
                grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
                grid[end_idx] = defender;
                result = CombatResult::DefenderWins;
            } else {
                grid[start_idx] = Piece{PieceType::Empty, Player::None, false};
                grid[end_idx] = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::BothDestroyed;
            }
        }
    }

    auto& my_hashes = (current_turn == Player::Red) ? red_chase_hashes : blue_chase_hashes;
    if (!defender.is_empty()) {
        my_hashes.clear();
    } else {
        my_hashes.push_back(get_board_state());
    }

    move_history.push_back(move);
    move_count++;

    current_turn = (current_turn == Player::Red) ? Player::Blue : Player::Red;
    
    if (result != CombatResult::FlagCaptured && move_count >= max_moves) {
        return CombatResult::Draw;
    }

    return result;
}

Piece Board::get_piece(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) return Piece{PieceType::Empty, Player::None, false};
    return grid[index(x, y)];
}


} // namespace stratego