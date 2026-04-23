#include "stratego/policy.h"

class NeuralPolicy : public Policy {
public:
    Move get_move(const GameState& masked_state) override {
        // TODO: Implement this function to run the masked_state through
        // your neural network and return the best move.
    }
    bool is_human() const override { return false; }
};