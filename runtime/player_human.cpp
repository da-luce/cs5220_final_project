#include "player_human.h"

namespace stratego {

void Human::set_next_move(const Move& m) {
    next_move = m;
}

Move Human::get_move(const GameState& masked_state) {
    if (next_move) {
        Move m = *next_move;
        next_move.reset();
        return m;
    }
    return Move{0, 0, 0, 0}; // Fallback: shouldn't be hit if orchestrated safely
}

bool Human::is_human() const {
    return true;
}

bool Human::has_move_ready() const {
    return next_move.has_value();
}

} // namespace stratego