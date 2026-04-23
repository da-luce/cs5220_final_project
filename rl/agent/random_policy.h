#pragma once
#include "stratego/policy.h"
#include "stratego/engine.h"
#include <vector>
#include <random>

class RandomPolicy : public stratego::Policy {
public:
    stratego::Move get_move(const stratego::GameState& masked_state) override {
        auto legal_moves = stratego::Engine::get_all_legal_moves(masked_state, masked_state.current_turn);
        if (legal_moves.empty()) {
            return {0, 0, 0, 0}; // Should not happen in a valid game state
        }
        std::uniform_int_distribution<size_t> dist(0, legal_moves.size() - 1);
        return legal_moves[dist(rng)];
    }
    bool is_human() const override { return false; }

private:
    std::mt19937 rng{std::random_device{}()};
};
