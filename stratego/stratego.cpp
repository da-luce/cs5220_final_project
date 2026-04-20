#include "stratego.h"
#include "setup_generator.h"
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

void Board::initialize_game(GameType type) {
    if (type == GameType::Normal) {
        width = 10;
        height = 10;
        initialize_empty();

        try {
            // The path is relative to the build directory where the executable runs.
            auto distributions = stratego::setup::load_distributions_from_json("../data/piece_dist.json");
            
            // Generate setups for both players using the new probabilistic generator.
            stratego::setup::generate_probabilistic_setup(*this, Player::Red, distributions);
            stratego::setup::generate_probabilistic_setup(*this, Player::Blue, distributions);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not load probabilistic setup from JSON (" << e.what() << ").\n"
                      << "Falling back to random setup." << std::endl;
            
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
        }
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
    if (move.start_x < 0 || move.start_x >= width || move.start_y < 0 || move.start_y >= height) return false;
    if (move.end_x < 0 || move.end_x >= width || move.end_y < 0 || move.end_y >= height) return false;

    Piece start_piece = grid[index(move.start_x, move.start_y)];
    if (start_piece.owner != current_turn) return false;
    if (!start_piece.is_mobile()) return false;

    Piece end_piece = grid[index(move.end_x, move.end_y)];
    if (end_piece.owner == current_turn) return false;
    if (end_piece.is_obstacle()) return false;

    int dx = move.end_x - move.start_x;
    int dy = move.end_y - move.start_y;

    if (dx != 0 && dy != 0) return false; // Must be straight line
    if (dx == 0 && dy == 0) return false; // Must actually move

    int dist = std::max(std::abs(dx), std::abs(dy));

    if (start_piece.type != PieceType::Scout && dist > 1) return false; // Only scouts can move > 1

    // Check for obstacles in the path (crucial for scouts moving > 1)
    int step_x = (dx == 0) ? 0 : (dx > 0 ? 1 : -1);
    int step_y = (dy == 0) ? 0 : (dy > 0 ? 1 : -1);

    for (int i = 1; i < dist; ++i) {
        int cx = move.start_x + i * step_x;
        int cy = move.start_y + i * step_y;
        if (!grid[index(cx, cy)].is_empty()) {
            return false;
        }
    }

    // Rule 10: Two-Squares Rule
    if (move_history.size() >= 6) {
        bool all_same = true;
        for (int i = 1; i <= 3; ++i) {
            const Move& prev = move_history[move_history.size() - i * 2];
            bool same = (prev.start_x == move.start_x && prev.start_y == move.start_y &&
                         prev.end_x == move.end_x && prev.end_y == move.end_y);
            bool reverse = (prev.start_x == move.end_x && prev.start_y == move.end_y &&
                            prev.end_x == move.start_x && prev.end_y == move.start_y);
            if (!same && !reverse) {
                all_same = false;
                break;
            }
        }
        if (all_same) return false;
    }

    // Rule 11: More-Squares Rule
    bool is_reverse_of_last = false;
    if (move_history.size() >= 2) {
        const Move& my_last = move_history[move_history.size() - 2];
        if (move.start_x == my_last.end_x && move.start_y == my_last.end_y &&
            move.end_x == my_last.start_x && move.end_y == my_last.start_y) {
            is_reverse_of_last = true;
        }
    }

    const auto& my_hashes = (current_turn == Player::Red) ? red_chase_hashes : blue_chase_hashes;
    if (!my_hashes.empty() && end_piece.is_empty() && !is_reverse_of_last) {
        Board temp = *this;
        temp.grid[temp.index(move.end_x, move.end_y)] = temp.grid[temp.index(move.start_x, move.start_y)];
        temp.grid[temp.index(move.start_x, move.start_y)] = Piece{PieceType::Empty, Player::None, false};
        std::string new_state = temp.get_board_state();

        if (std::find(my_hashes.begin(), my_hashes.end(), new_state) != my_hashes.end()) {
            return false;
        }
    }

    return true;
}

CombatResult Board::execute_move(const Move& move) {
    if (!is_legal_move(move)) return CombatResult::InvalidMove;

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

std::vector<Move> Board::get_all_legal_moves(Player player) const {
    std::vector<Move> moves;
    if (player != current_turn) return moves;

    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (grid[index(x, y)].owner == player) {
                Piece p = grid[index(x, y)];
                if (!p.is_mobile()) continue;
                
                int max_dist = (p.type == PieceType::Scout) ? std::max(width, height) : 1;

                for (int dir = 0; dir < 4; ++dir) {
                    for (int dist = 1; dist <= max_dist; ++dist) {
                        Move m{x, y, x + dx[dir] * dist, y + dy[dir] * dist};
                        if (is_legal_move(m)) {
                            moves.push_back(m);
                        } else {
                            break; // Stop looking further in this direction if an obstacle is hit
                        }
                    }
                }
            }
        }
    }
    return moves;
}

std::vector<Move> Board::get_legal_moves_for_piece(Player player, int x, int y) const {
    std::vector<Move> moves;

    // Must be the player's turn, and coordinates must be within bounds
    if (player != current_turn) return moves;
    if (x < 0 || x >= width || y < 0 || y >= height) return moves;

    Piece p = grid[index(x, y)];
    
    // The piece must belong to the requesting player and be mobile (not a flag/bomb/empty)
    if (p.owner != player || !p.is_mobile()) return moves;

    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    int max_dist = (p.type == PieceType::Scout) ? std::max(width, height) : 1;

    for (int dir = 0; dir < 4; ++dir) {
        for (int dist = 1; dist <= max_dist; ++dist) {
            Move m{x, y, x + dx[dir] * dist, y + dy[dir] * dist};
            
            // is_legal_move handles bounds, obstacles, and friendly fire checks
            if (is_legal_move(m)) {
                moves.push_back(m);
            } else {
                // For scouts, if a move is invalid (e.g., hits an obstacle or board edge), 
                // all further moves in this direction are also invalid.
                break; 
            }
        }
    }

    return moves;
}

bool Board::save_to_file(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }

    // 1. Write metadata (dimensions, current turn)
    ofs.write(reinterpret_cast<const char*>(&width), sizeof(width));
    ofs.write(reinterpret_cast<const char*>(&height), sizeof(height));
    char turn = static_cast<char>(current_turn);
    ofs.write(&turn, sizeof(turn));

    // 2. Write grid data piece by piece
    for (const auto& piece : grid) {
        char type = static_cast<char>(piece.type);
        char owner = static_cast<char>(piece.owner);
        char revealed = piece.revealed ? 1 : 0;
        ofs.write(&type, sizeof(type));
        ofs.write(&owner, sizeof(owner));
        ofs.write(&revealed, sizeof(revealed));
    }

    return ofs.good();
}

bool Board::load_from_file(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        return false;
    }

    // 1. Read metadata
    int new_width, new_height;
    ifs.read(reinterpret_cast<char*>(&new_width), sizeof(new_width));
    ifs.read(reinterpret_cast<char*>(&new_height), sizeof(new_height));
    char turn;
    ifs.read(&turn, sizeof(turn));

    if (ifs.fail()) return false;

    // 2. Update board state from metadata
    width = new_width;
    height = new_height;
    current_turn = static_cast<Player>(turn);
    move_count = 0;
    move_history.clear();
    red_chase_hashes.clear();
    blue_chase_hashes.clear();
    grid.resize(width * height);

    // 3. Read grid data
    for (auto& piece : grid) {
        char type_c, owner_c, revealed_c;
        ifs.read(&type_c, sizeof(type_c));
        ifs.read(&owner_c, sizeof(owner_c));
        ifs.read(&revealed_c, sizeof(revealed_c));

        if (ifs.fail()) return false;

        piece = {static_cast<PieceType>(type_c), static_cast<Player>(owner_c), (revealed_c != 0)};
    }

    return ifs.peek() == EOF; // Ensure we successfully read the whole file
}

} // namespace stratego