#pragma once
#include "../stratego/state.h"
#include "player_agent.h"
#include <optional>

namespace stratego {

class Human : public Policy {
private:
    std::optional<Move> next_move;

public:
    void set_next_move(const Move& m);
    
    // Consumes the provided move. If none is available, returns a fallback move.
    // The orchestrator must be coordinated with the UI loop to only step when a move is ready.
    Move get_move(const GameState& masked_state) override;

    bool is_human() const override;
    
    bool has_move_ready() const;
};

} // namespace stratego