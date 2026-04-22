#pragma once
#include "board.h"
#include <vector>
#include <string>

namespace stratego {

struct Move {
    int start_x, start_y;
    int end_x, end_y;
};

enum class CombatResult {
    MovedToEmpty,
    AttackerWins,
    DefenderWins,
    BothDestroyed,
    FlagCaptured,
    InvalidMove,
    Draw
};

class Engine {
public:
    // The last two parameters are used for enforcing the More-Squares Rule,
    // which prevents infinite loops of non-revealing moves.
    static bool is_legal_move(
        const Board& board,
        const Move& move,
        Player current_player,
        const std::vector<GameHash>& chase_hashes,
        const std::vector<Move>& history
    );
    static std::vector<Move> get_all_legal_moves(
        const Board& board,
        Player player,
        const std::vector<GameHash>& chase_hashes,
        const std::vector<Move>& history
    );
    // x, y refers to the piece's starting position
    static std::vector<Move> get_legal_moves_for_piece(
        const Board& board, 
        int x, 
        int y,
        const std::vector<GameHash>& chase_hashes,
        const std::vector<Move>& history
    );

    // Modifies the board and returns the outcome of the action
    static CombatResult execute_move(Board& board, const Move& move);

    // Utility for flipping moves when generating opponent views
    static Move get_flipped_move(const BoardConfig& config, const Move& move);
};

} // namespace stratego