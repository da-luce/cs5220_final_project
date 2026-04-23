#pragma once
#include "state.h"

namespace stratego {

class Policy {
public:
    virtual ~Policy() = default;

    // Returns a Move. Bots calculate it; Humans return a move stored from a UI click.
    virtual Move get_move(const GameState& masked_state) = 0;
    virtual bool is_human() const = 0;
};

}
