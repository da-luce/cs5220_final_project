#include "stratego.h"
#include <cmath>
#include <algorithm>

namespace stratego {

Board::Board(int w, int h) : width(w), height(h), grid(w * h), current_turn(Player::Red) {
    initialize_empty();
}

void Board::initialize_empty() {
    grid.assign(width * height, Piece{PieceType::Empty, Player::None, false});
    current_turn = Player::Red;

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
    }
}

bool Board::place_piece(int x, int y, PieceType type, Player owner) {
    if (x < 0 || x >= width || y < 0 || y >= height) return false;
    int idx = index(x, y);
    if (!grid[idx].is_empty()) return false;
    
    grid[idx] = Piece{type, owner, false};
    return true;
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

    return true;
}

CombatResult Board::execute_move(const Move& move) {
    if (!is_legal_move(move)) return CombatResult::InvalidMove;

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

    current_turn = (current_turn == Player::Red) ? Player::Blue : Player::Red;
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

} // namespace stratego