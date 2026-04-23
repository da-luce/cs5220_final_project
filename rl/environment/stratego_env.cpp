#include "stratego_env.h"

StrategoEnvironment::StrategoEnvironment(
    stratego::BoardConfig config, 
    stratego::state::SetupType setup_type,
    int max_moves
) : config(config), setup_type(setup_type), 
    max_moves(max_moves), action_encoder(config) {
    reset();
}

torch::Tensor StrategoEnvironment::reset() {
    state = stratego::state::initialize(config, setup_type, max_moves);
    return get_current_obs();
}

StepResult<torch::Tensor> StrategoEnvironment::step(const int& action_idx) {
    // 1. Python-style Action Indexing: (y * 4 + x) * 4 + direction
    int y = (action_idx / 4) / 4;
    int x = (action_idx / 4) % 4;
    int d = action_idx % 4;

    // 2. Python-style directions: 0:Up, 1:Down, 2:Left, 3:Right (in VIEW space)
    int dy = (d == 0) ? -1 : (d == 1 ? 1 : 0);
    int dx = (d == 2) ? -1 : (d == 3 ? 1 : 0);
    
    stratego::Move view_move{x, y, x + dx, y + dy};
    
    // 3. Transform to Global Space
    stratego::Move global_move = (state.current_turn == stratego::Player::Red) ? 
                                 view_move : stratego::Engine::get_flipped_move(config, view_move);
    
    StepResult<torch::Tensor> result;
    stratego::CombatResult outcome = stratego::Engine::execute_move(state, global_move);
    
    if (outcome == stratego::CombatResult::FlagCaptured) {
        result.reward = 1.0f;
        result.terminated = true;
    } else if (outcome == stratego::CombatResult::InvalidMove) {
        result.reward = -1.0f;
        result.terminated = true;
    } else if (state.move_count >= state.max_moves) {
        result.reward = 0.0f;
        result.terminated = true;
    } else {
        if (state.move_count > 10 && stratego::Engine::get_all_legal_moves(state, state.current_turn).empty()) {
            result.reward = 1.0f;
            result.terminated = true;
        }
    }

    result.observation = get_current_obs();
    return result;
}

torch::Tensor StrategoEnvironment::get_current_obs() const {
    // Python order: Flag, Lieut(5), Capt(6), Major(7)
    int C = 9; // Python parity
    int H = 4, W = 4;
    auto obs = torch::zeros({C, H, W}, torch::kFloat32);
    auto accessor = obs.accessor<float, 3>();

    stratego::Player p = state.current_turn;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto piece = state.board.get_piece(x, y);
            if (piece.is_empty()) continue;

            int vx = (p == stratego::Player::Red) ? x : (W - 1 - x);
            int vy = (p == stratego::Player::Red) ? y : (H - 1 - y);

            int type_idx = -1;
            if (piece.type == stratego::PieceType::Flag) type_idx = 0;
            else if (piece.type == stratego::PieceType::Lieutenant) type_idx = 1;
            else if (piece.type == stratego::PieceType::Captain) type_idx = 2;
            else if (piece.type == stratego::PieceType::Major) type_idx = 3;

            if (piece.is_obstacle()) continue; // Tiny has no lakes

            if (piece.owner == p) {
                if (type_idx != -1) accessor[type_idx][vy][vx] = 1.0f;
            } else if (piece.revealed) {
                if (type_idx != -1) accessor[5 + type_idx][vy][vx] = 1.0f;
            } else {
                accessor[4][vy][vx] = 1.0f; // Hidden
            }
        }
    }
    return obs;
}

size_t StrategoEnvironment::observation_dim() const { return 9 * 4 * 4; }
size_t StrategoEnvironment::action_dim() const { return 64; }

torch::Tensor StrategoEnvironment::get_action_mask() const {
    auto mask = torch::zeros({64}, torch::kFloat32);
    auto accessor = mask.accessor<float, 1>();
    auto legal_moves = stratego::Engine::get_all_legal_moves(state, state.current_turn);
    
    for (const auto& m : legal_moves) {
        stratego::Move view_m = (state.current_turn == stratego::Player::Red) ? 
                                 m : stratego::Engine::get_flipped_move(config, m);
        
        int dx = view_m.end_x - view_m.start_x;
        int dy = view_m.end_y - view_m.start_y;
        
        // Only 1-square moves are encoded in the 64-action space
        if (std::abs(dx) + std::abs(dy) == 1) {
            int d = (dy == -1) ? 0 : (dy == 1 ? 1 : (dx == -1 ? 2 : 3));
            int action_idx = (view_m.start_y * 4 + view_m.start_x) * 4 + d;
            accessor[action_idx] = 1.0f;
        }
    }
    return mask;
}
