namespace stratego {

class RandomBot : public PlayerAgent {
public:
    Move get_move(const GameState& masked_state) override {
        auto legal_moves = Engine::get_all_legal_moves(masked_state, masked_state.current_turn);
        // Pick one and return it
        return legal_moves[rand() % legal_moves.size()];
    }
    bool is_human() const override { return false; }
};

} // namespace stratego