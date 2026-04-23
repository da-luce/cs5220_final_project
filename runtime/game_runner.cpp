#pragma once
#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../policy.h"
#include <memory>

namespace stratego {

class GameRunner {
private:
    GameState root_state;
    std::unique_ptr<Policy> red_player;
    std::unique_ptr<Policy> blue_player;

public:
    GameRunner(GameState initial_state, 
                 std::unique_ptr<Policy> red, 
                 std::unique_ptr<Policy> blue)
        : root_state(std::move(initial_state)), 
          red_player(std::move(red)), 
          blue_player(std::move(blue)) {}

    CombatResult step() {
        Player current = root_state.current_turn;
        PlayerAgent* active_agent = (current == Player::Red) ? red_player.get() : blue_player.get();

        // 1. Prepare the view for the agent
        GameState view = state::get_masked_view(root_state, current);

        // 2. Get move from agent
        Move move = active_agent->get_move(view);

        // 3. Update the real game
        return Engine::execute_move(root_state, move);
    }
    
    const GameState& get_state() const { return root_state; }
    PlayerAgent* get_active_agent() const { return (root_state.current_turn == Player::Red) ? red_player.get() : blue_player.get(); }
};

} // namespace stratego