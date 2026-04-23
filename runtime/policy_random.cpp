#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../stratego/policy.h"
#include <vector>
#include <cstdlib>

namespace stratego {

class Random : public Policy {
public:
    Move get_move(const GameState& masked_state) override {
        auto legal_moves = Engine::get_all_legal_moves(masked_state, masked_state.current_turn);
        if (legal_moves.empty()) return {0,0,0,0};
        // Pick one and return it
        return legal_moves[rand() % legal_moves.size()];
    }
    bool is_human() const override { return false; }
};

} // namespace stratego
