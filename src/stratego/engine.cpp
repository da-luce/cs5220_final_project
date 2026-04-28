#include "engine.h" // Replace with your actual header name
#include <algorithm>
#include <cmath>

namespace stratego {

bool Engine::is_legal_move(
    const GameState& state, 
    const Move& move) 
{
    const Board& board = state.board;
    Player current_player = state.current_turn;
    const std::vector<GameHash>& chase_hashes = state.chase_hashes;
    const std::vector<Move>& history = state.move_history;

    // 1. Boundary Checks
    if (move.start_x < 0 || move.start_x >= board.get_width() || move.start_y < 0 || move.start_y >= board.get_height()) return false;
    if (move.end_x < 0 || move.end_x >= board.get_width() || move.end_y < 0 || move.end_y >= board.get_height()) return false;

    // 2. Ownership and Mobility Checks
    Piece start_piece = board.grid[board.index(move.start_x, move.start_y)];
    if (start_piece.owner != current_player) return false;
    if (!start_piece.is_mobile()) return false;

    Piece end_piece = board.grid[board.index(move.end_x, move.end_y)];
    if (end_piece.owner == current_player) return false;
    if (end_piece.is_obstacle()) return false;

    // 3. Movement Vector Checks
    int dx = move.end_x - move.start_x;
    int dy = move.end_y - move.start_y;

    if (dx != 0 && dy != 0) return false; // Must be straight line
    if (dx == 0 && dy == 0) return false; // Must actually move

    int dist = std::max(std::abs(dx), std::abs(dy));
    if (start_piece.type != PieceType::Scout && dist > 1) return false; // Only scouts can move > 1

    // 4. Obstacle Collision Checks
    int step_x = (dx == 0) ? 0 : (dx > 0 ? 1 : -1);
    int step_y = (dy == 0) ? 0 : (dy > 0 ? 1 : -1);

    for (int i = 1; i < dist; ++i) {
        int cx = move.start_x + i * step_x;
        int cy = move.start_y + i * step_y;
        if (!board.grid[board.index(cx, cy)].is_empty()) {
            return false; // Path is blocked
        }
    }

    // 5. Rule: Two-Squares Rule
    if (board.get_width() >= 5 && history.size() >= 6) {
        bool all_same = true;
        // Check our last 3 moves (which are at index -2, -4, -6 due to turn alternation)
        for (int i = 1; i <= 3; ++i) {
            const Move& prev = history[history.size() - (i * 2)];
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

    // 6. Rule: More-Squares Rule (Repetition Prevention)
    bool is_reverse_of_last = false;
    if (history.size() >= 2) {
        const Move& my_last = history[history.size() - 2];
        if (move.start_x == my_last.end_x && move.start_y == my_last.end_y &&
            move.end_x == my_last.start_x && move.end_y == my_last.start_y) {
            is_reverse_of_last = true;
        }
    }

    // Only non-attacking moves trigger the chase rule
    if (!chase_hashes.empty() && end_piece.is_empty() && !is_reverse_of_last) {
        Board temp = board; // Copy board to simulate the move
        temp.grid[temp.index(move.end_x, move.end_y)] = temp.grid[temp.index(move.start_x, move.start_y)];
        temp.grid[temp.index(move.start_x, move.start_y)] = Piece{PieceType::Empty, Player::None, false};
        
        // Calculate the hash of the resulting board. 
        // Note: The resulting state becomes the opponent's turn.
        Player next_player = (current_player == Player::Red) ? Player::Blue : Player::Red;
        GameHash new_state = temp.compute_hash(next_player);

        if (std::find(chase_hashes.begin(), chase_hashes.end(), new_state) != chase_hashes.end()) {
            return false;
        }
    }

    return true;
}

std::vector<Move> Engine::get_legal_moves_for_piece(
    const GameState& state, 
    int x, int y) 
{
    const Board& board = state.board;
    std::vector<Move> moves;

    if (x < 0 || x >= board.get_width() || y < 0 || y >= board.get_height()) return moves;

    Piece p = board.grid[board.index(x, y)];
    if (!p.is_mobile() || p.owner == Player::None) return moves;

    Player player = p.owner;
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    int max_dist = (p.type == PieceType::Scout) ? std::max(board.get_width(), board.get_height()) : 1;

    for (int dir = 0; dir < 4; ++dir) {
        for (int dist = 1; dist <= max_dist; ++dist) {
            int nx = x + dx[dir] * dist;
            int ny = y + dy[dir] * dist;

            // Bounds check
            if (nx < 0 || nx >= board.get_width() || ny < 0 || ny >= board.get_height()) break;

            Piece target = board.grid[board.index(nx, ny)];

            // Stop sliding entirely if we hit a friendly piece or an obstacle
            if (target.is_obstacle() || target.owner == player) break;

            Move m{x, y, nx, ny};
            
            // Check if this specific destination is legal (handles repetition rules)
            if (is_legal_move(state, m)) {
                moves.push_back(m);
            }

            // If we hit an enemy piece, we can attack it, but a Scout cannot slide PAST it
            if (target.owner != player && target.owner != Player::None) break;
        }
    }

    return moves;
}

std::vector<Move> Engine::get_all_legal_moves(
    const GameState& state, 
    Player player) 
{
    const Board& board = state.board;
    std::vector<Move> all_moves;

    for (int y = 0; y < board.get_height(); ++y) {
        for (int x = 0; x < board.get_width(); ++x) {
            if (board.grid[board.index(x, y)].owner == player) {
                std::vector<Move> piece_moves = get_legal_moves_for_piece(state, x, y);
                all_moves.insert(all_moves.end(), piece_moves.begin(), piece_moves.end());
            }
        }
    }

    return all_moves;
}

CombatResult Engine::execute_move(GameState& state, const Move& move) {

    // Check max move limit before executing
    if (state.move_count >= state.max_moves) {
        return CombatResult::Draw;
    }

    // Check legality
    if (!is_legal_move(state, move)) {
        return CombatResult::InvalidMove;
    }

    Board& board = state.board;
    int start_idx = board.index(move.start_x, move.start_y);
    int end_idx = board.index(move.end_x, move.end_y);

    Piece& attacker = board.grid[start_idx];
    Piece& defender = board.grid[end_idx];

    // Capture true identities before the squares get overwritten below.
    PieceType true_attacker_type = attacker.type;
    PieceType true_defender_type = defender.type;

    CombatResult result;

    // 1. Moving to an empty space
    if (defender.is_empty()) {
        defender = attacker;
        attacker = Piece{PieceType::Empty, Player::None, false};
        result = CombatResult::MovedToEmpty;
    }
    else {
        // 2. Combat Resolution
        attacker.revealed = true;
        defender.revealed = true;

        if (defender.type == PieceType::Flag) {
            defender = attacker;
            attacker = Piece{PieceType::Empty, Player::None, false};
            result = CombatResult::FlagCaptured;
        }
        else if (defender.type == PieceType::Bomb) {
            if (attacker.type == PieceType::Miner) {
                defender = attacker; // Miner defuses bomb
                attacker = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::AttackerWins;
            } else {
                attacker = Piece{PieceType::Empty, Player::None, false}; // Attacker blows up
                result = CombatResult::DefenderWins;
            }
        }
        else if (defender.type == PieceType::Marshal && attacker.type == PieceType::Spy) {
            defender = attacker; // Spy kills Marshal when attacking
            attacker = Piece{PieceType::Empty, Player::None, false};
            result = CombatResult::AttackerWins;
        }
        else {
            // Standard Rank Comparison (Higher integer value wins)
            int attacker_val = static_cast<int>(attacker.type);
            int defender_val = static_cast<int>(defender.type);

            if (attacker_val > defender_val) {
                defender = attacker;
                attacker = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::AttackerWins;
            } 
            else if (attacker_val < defender_val) {
                attacker = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::DefenderWins;
            } 
            else {
                // Tie - Both destroyed
                attacker = Piece{PieceType::Empty, Player::None, false};
                defender = Piece{PieceType::Empty, Player::None, false};
                result = CombatResult::BothDestroyed;
            }
        }
    }

    // Update match history. Only record piece identities that were publicly
    // revealed this turn — a quiet move to an empty square must not leak the
    // mover's rank into the shared history (Fog of War).
    Move recorded = move;
    if (result == CombatResult::MovedToEmpty) {
        recorded.attacker_type = PieceType::Empty;
        recorded.defender_type = PieceType::Empty;
    } else {
        recorded.attacker_type = true_attacker_type;
        recorded.defender_type = true_defender_type;
    }
    state.move_history.push_back(recorded);
    state.move_count++;
    
    // Swap the turn to the opponent
    state.current_turn = (state.current_turn == Player::Red) ? Player::Blue : Player::Red;
    
    // Log the resulting hash for repetition checking (More-Squares Rule)
    state.chase_hashes.push_back(state.board.compute_hash(state.current_turn));

    // Match-Level Limits
    if (state.max_moves > 0 && state.move_count >= state.max_moves) {
        return CombatResult::Draw;
    }

    return result;
}

Move Engine::get_flipped_move(const BoardConfig& config, const Move& move) {
    return Move{
        config.width - 1 - move.start_x,
        config.height - 1 - move.start_y,
        config.width - 1 - move.end_x,
        config.height - 1 - move.end_y,
        move.attacker_type,
        move.defender_type
    };
}

} // namespace stratego