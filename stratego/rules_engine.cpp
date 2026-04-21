#include "rules_engine.h"
#include <algorithm>
#include <cmath>

namespace stratego {

bool RulesEngine::is_legal_move(const Board& board, const Move& move) {
    if (move.start_x < 0 || move.start_x >= board.get_width() || move.start_y < 0 || move.start_y >= board.get_height()) return false;
    if (move.end_x < 0 || move.end_x >= board.get_width() || move.end_y < 0 || move.end_y >= board.get_height()) return false;

    Piece start_piece = board.grid[board.index(move.start_x, move.start_y)];
    if (start_piece.owner != board.current_turn) return false;
    if (!start_piece.is_mobile()) return false;

    Piece end_piece = board.grid[board.index(move.end_x, move.end_y)];
    if (end_piece.owner == board.current_turn) return false;
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
        if (!board.grid[board.index(cx, cy)].is_empty()) {
            return false;
        }
    }

    // Rule 10: Two-Squares Rule
    if (board.move_history.size() >= 6) {
        bool all_same = true;
        for (int i = 1; i <= 3; ++i) {
            const Move& prev = board.move_history[board.move_history.size() - i * 2];
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
    if (board.move_history.size() >= 2) {
        const Move& my_last = board.move_history[board.move_history.size() - 2];
        if (move.start_x == my_last.end_x && move.start_y == my_last.end_y &&
            move.end_x == my_last.start_x && move.end_y == my_last.start_y) {
            is_reverse_of_last = true;
        }
    }

    const auto& my_hashes = (board.current_turn == Player::Red) ? board.red_chase_hashes : board.blue_chase_hashes;
    if (!my_hashes.empty() && end_piece.is_empty() && !is_reverse_of_last) {
        Board temp = board;
        temp.grid[temp.index(move.end_x, move.end_y)] = temp.grid[temp.index(move.start_x, move.start_y)];
        temp.grid[temp.index(move.start_x, move.start_y)] = Piece{PieceType::Empty, Player::None, false};
        std::string new_state = temp.get_board_state();

        if (std::find(my_hashes.begin(), my_hashes.end(), new_state) != my_hashes.end()) {
            return false;
        }
    }

    return true;
}

std::vector<Move> RulesEngine::get_all_legal_moves(const Board& board, Player player) {
    std::vector<Move> moves;
    if (player != board.current_turn) return moves;

    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    for (int y = 0; y < board.get_height(); ++y) {
        for (int x = 0; x < board.get_width(); ++x) {
            if (board.grid[board.index(x, y)].owner == player) {
                Piece p = board.grid[board.index(x, y)];
                if (!p.is_mobile()) continue;
                
                int max_dist = (p.type == PieceType::Scout) ? std::max(board.get_width(), board.get_height()) : 1;

                for (int dir = 0; dir < 4; ++dir) {
                    for (int dist = 1; dist <= max_dist; ++dist) {
                        Move m{x, y, x + dx[dir] * dist, y + dy[dir] * dist};
                        if (is_legal_move(board, m)) {
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

std::vector<Move> RulesEngine::get_legal_moves_for_piece(const Board& board, Player player, int x, int y) {
    std::vector<Move> moves;

    if (player != board.current_turn) return moves;
    if (x < 0 || x >= board.get_width() || y < 0 || y >= board.get_height()) return moves;

    Piece p = board.grid[board.index(x, y)];
    if (p.owner != player || !p.is_mobile()) return moves;

    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    int max_dist = (p.type == PieceType::Scout) ? std::max(board.get_width(), board.get_height()) : 1;

    for (int dir = 0; dir < 4; ++dir) {
        for (int dist = 1; dist <= max_dist; ++dist) {
            Move m{x, y, x + dx[dir] * dist, y + dy[dir] * dist};
            
            if (is_legal_move(board, m)) {
                moves.push_back(m);
            } else {
                break; 
            }
        }
    }

    return moves;
}

} // namespace stratego