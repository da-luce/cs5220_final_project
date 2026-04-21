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

Board::Board(const GameConfig& cfg) : config(cfg), grid(cfg.width * cfg.height), current_turn(Player::Red), move_count(0) {
    clear();
}

void Board::clear() {
    grid.assign(config.width * config.height, Piece{PieceType::Empty, Player::None, false});
    current_turn = Player::Red;
    move_count = 0;
    move_history.clear();
    red_chase_hashes.clear();
    blue_chase_hashes.clear();

    for (const auto& lake : config.lakes) {
        for (int y = lake.y; y < lake.y + lake.h; ++y) {
            for (int x = lake.x; x < lake.x + lake.w; ++x) {
                grid[index(x, y)] = Piece{PieceType::Water, Player::None, true};
            }
        }
    }
}

GameConfig get_config_for_game_type(GameType type) {
    if (type == GameType::Classic) {
        return {10, 10, 4, 2000, {
            {PieceType::Flag, 1}, {PieceType::Marshal, 1}, {PieceType::General, 1}, {PieceType::Spy, 1},
            {PieceType::Bomb, 6}, {PieceType::Scout, 8}, {PieceType::Miner, 5},
            {PieceType::Sergeant, 4}, {PieceType::Lieutenant, 4}, {PieceType::Captain, 4},
            {PieceType::Major, 3}, {PieceType::Colonel, 2}
        },
        {{2, 4, 2, 2}, {6, 4, 2, 2}}};
    } else if (type == GameType::Tiny) {
        return {4, 4, 1, 2000, {
            {PieceType::Flag, 1}, {PieceType::Major, 1}, {PieceType::Captain, 1}, {PieceType::Lieutenant, 1}
        },
        {}};
    } else if (type == GameType::Quick) {
        return {8, 8, 2, 2000, {
            {PieceType::Flag, 1}, {PieceType::Spy, 1}, {PieceType::Marshal, 1}, {PieceType::General, 1},
            {PieceType::Bomb, 2}, {PieceType::Miner, 2}, {PieceType::Scout, 2}
        },
        {{2, 3, 1, 2}, {5, 3, 1, 2}}};
    } else if (type == GameType::Barrage) {
        return {10, 10, 4, 2000, {
            {PieceType::Flag, 1}, {PieceType::Marshal, 1}, {PieceType::General, 1},
            {PieceType::Miner, 1}, {PieceType::Spy, 1}, {PieceType::Bomb, 1},
            {PieceType::Scout, 2}
        },
        {{2, 4, 2, 2}, {6, 4, 2, 2}}};
    }
    throw std::invalid_argument("Unknown GameType");
}

void Board::initialize_game(SetupType setup) {
    clear();

    bool use_probabilistic = (setup == SetupType::Probabilistic || setup == SetupType::Default) && config.width == 10 && config.height == 10;

    if (use_probabilistic) {
        try {
            // The path is relative to the build directory where the executable runs.
            auto distributions = stratego::setup::load_distributions_from_json("../data/piece_dist.json");
            
            // Generate setups for both players using the new probabilistic generator.
            stratego::setup::generate_probabilistic_setup(*this, Player::Red, config.piece_counts, distributions);
            stratego::setup::generate_probabilistic_setup(*this, Player::Blue, config.piece_counts, distributions);
            return;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not load probabilistic setup from JSON (" << e.what() << ").\n"
                      << "Falling back to random setup." << std::endl;
        }
    }
    
    // Fallback to the original random setup if the JSON is missing or invalid.
    auto setup_army = [&](Player player, int start_y) {
            std::vector<PieceType> deck;
            for (const auto& pair : config.piece_counts) {
                deck.insert(deck.end(), pair.second, pair.first);
            }

            int total_pieces = deck.size();
            int setup_area_size = config.width * config.setup_rows;
            if (total_pieces < setup_area_size) {
                deck.insert(deck.end(), setup_area_size - total_pieces, PieceType::Empty);
            }

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(deck.begin(), deck.end(), g);

            int idx = 0;
            for (int y = start_y; y < start_y + config.setup_rows; ++y) {
                for (int x = 0; x < config.width; ++x) {
                    if (idx < deck.size() && deck[idx] != PieceType::Empty) {
                        place_piece(x, y, deck[idx], player);
                    }
                    idx++;
                }
            }
        };
        setup_army(Player::Blue, 0);
        setup_army(Player::Red, config.height - config.setup_rows);
}

bool Board::place_piece(int x, int y, PieceType type, Player owner) {
    if (x < 0 || x >= config.width || y < 0 || y >= config.height) return false;
    int idx = index(x, y);
    if (!grid[idx].is_empty()) return false;
    
    grid[idx] = Piece{type, owner, false};
    return true;
}

std::string Board::get_board_state() const {
    std::string state;
    state.reserve(config.width * config.height * 2);
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
    
    if (result != CombatResult::FlagCaptured && move_count >= config.max_moves) {
        return CombatResult::Draw;
    }

    return result;
}

Piece Board::get_piece(int x, int y) const {
    if (x < 0 || x >= config.width || y < 0 || y >= config.height) return Piece{PieceType::Empty, Player::None, false};
    return grid[index(x, y)];
}


} // namespace stratego