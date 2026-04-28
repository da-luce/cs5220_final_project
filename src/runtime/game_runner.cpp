#pragma once
#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../stratego/policy.h"
#include <memory>

namespace stratego {

class GameRunner {
private:
    GameState root_state;
    std::unique_ptr<Policy> red_player;
    std::unique_ptr<Policy> blue_player;
    Move last_attempted_move{};
    Player last_attempted_player{Player::None};

public:
    GameRunner(GameState initial_state,
                 std::unique_ptr<Policy> red,
                 std::unique_ptr<Policy> blue)
        : root_state(std::move(initial_state)),
          red_player(std::move(red)),
          blue_player(std::move(blue)) {}

    CombatResult step() {
        Player current = root_state.current_turn;

        // If the active player has no legal moves, they lose — don't bother
        // asking the policy (Random would just return a 0-move sentinel and
        // we'd report it as an "invalid move", which is misleading).
        if (Engine::get_all_legal_moves(root_state, current).empty()) {
            last_attempted_player = current;
            last_attempted_move = Move{};
            return CombatOutcome::NoLegalMoves;
        }

        Policy* active_agent = (current == Player::Red) ? red_player.get() : blue_player.get();

        // 1. Prepare the view for the agent
        GameState view = state::get_masked_view(root_state, current);

        // 2. Get move from agent
        Move move = active_agent->get_move(view);
        last_attempted_move = move;
        last_attempted_player = current;

        // 3. Update the real game
        return Engine::execute_move(root_state, move);
    }

    const GameState& get_state() const { return root_state; }
    Policy* get_active_agent() const { return (root_state.current_turn == Player::Red) ? red_player.get() : blue_player.get(); }
    const Move& get_last_attempted_move() const { return last_attempted_move; }
    Player get_last_attempted_player() const { return last_attempted_player; }
};

} // namespace stratego
