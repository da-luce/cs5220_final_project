#pragma once
#include "board.h"
#include <vector>
#include <string>
#include "state.h"

namespace stratego {

class Engine {
public:
    // The last two parameters are used for enforcing the More-Squares Rule,
    // which prevents infinite loops of non-revealing moves.
    static bool is_legal_move(
        const GameState& state,
        const Move& move
    );
    static std::vector<Move> get_all_legal_moves(
        const GameState& state,
        Player player
    );
    // x, y refers to the piece's starting position
    static std::vector<Move> get_legal_moves_for_piece(
        const GameState& state,
        int x,
        int y
    );

    // Modifies game state and returns the outcome of the action
    static CombatResult execute_move(GameState& state, const Move& move);

    // Utility for flipping moves when generating opponent views
    static Move get_flipped_move(const BoardConfig& config, const Move& move);
};

} // namespace stratego