#include "player_human.h"

namespace stratego {

void HumanPlayer::set_next_move(const Move& m) {
    next_move = m;
}

Move HumanPlayer::get_move(const GameState& masked_state) {
    if (next_move) {
        Move m = *next_move;
        next_move.reset();
        return m;
    }
    return Move{0, 0, 0, 0}; // Fallback: shouldn't be hit if orchestrated safely
}

bool HumanPlayer::is_human() const {
    return true;
}

bool HumanPlayer::has_move_ready() const {
    return next_move.has_value();
}

} // namespace stratego