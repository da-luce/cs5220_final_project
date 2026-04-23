#include "policy_neural.h"
#include "rl/encoding/board.h"
#include "stratego/engine.h"

NeuralPolicy::NeuralPolicy(const std::string& model_path, const stratego::BoardConfig& config)
    : config(config), action_encoder(config) {}

NeuralPolicy::NeuralPolicy(networks::StrategoNet model, const stratego::BoardConfig& config)
    : model(model), config(config), action_encoder(config) {}

stratego::Move NeuralPolicy::get_move(const stratego::GameState& masked_state) {
    torch::NoGradGuard no_grad;
    model->eval();

    // 1. Encode in VIEW SPACE using the EXACT same logic as training
    int C = 9; 
    int H = 4, W = 4;
    auto obs = torch::zeros({1, C, H, W}, torch::kFloat32);
    auto accessor = obs.accessor<float, 4>();
    stratego::Player p = masked_state.current_turn;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto piece = masked_state.board.get_piece(x, y);
            if (piece.is_empty()) continue;

            int vx = (p == stratego::Player::Red) ? x : (W - 1 - x);
            int vy = (p == stratego::Player::Red) ? y : (H - 1 - y);

            int type_idx = -1;
            if (piece.type == stratego::PieceType::Flag) type_idx = 0;
            else if (piece.type == stratego::PieceType::Lieutenant) type_idx = 1;
            else if (piece.type == stratego::PieceType::Captain) type_idx = 2;
            else if (piece.type == stratego::PieceType::Major) type_idx = 3;

            if (piece.owner == p) {
                if (type_idx != -1) accessor[0][type_idx][vy][vx] = 1.0f;
            } else if (piece.revealed) {
                if (type_idx != -1) accessor[0][5 + type_idx][vy][vx] = 1.0f;
            } else {
                accessor[0][4][vy][vx] = 1.0f; // Hidden channel (4)
            }
        }
    }

    // 2. Forward pass
    auto [logits, value] = model->forward(obs);
    
    // 3. Apply mask in VIEW SPACE
    torch::Tensor mask = get_action_mask(masked_state);
    logits.masked_fill_(mask.unsqueeze(0) == 0, -1e9);

    // 4. Select best action index
    int action_idx = torch::argmax(logits, 1).item<int>();
    
    // 5. Decode action index back to global move
    int vy = (action_idx / 4) / 4;
    int vx = (action_idx / 4) % 4;
    int d = action_idx % 4;
    int vdy = (d == 0) ? -1 : (d == 1 ? 1 : 0);
    int vdx = (d == 2) ? -1 : (d == 3 ? 1 : 0);
    
    stratego::Move view_move{vx, vy, vx + vdx, vy + vdy};
    return (p == stratego::Player::Red) ? view_move : stratego::Engine::get_flipped_move(config, view_move);
}

torch::Tensor NeuralPolicy::get_action_mask(const stratego::GameState& state) {
    auto mask = torch::zeros({64}, torch::kFloat32);
    auto accessor = mask.accessor<float, 1>();
    auto legal_moves = stratego::Engine::get_all_legal_moves(state, state.current_turn);
    
    for (const auto& m : legal_moves) {
        stratego::Move view_m = (state.current_turn == stratego::Player::Red) ? 
                                 m : stratego::Engine::get_flipped_move(config, m);
        int dx = view_m.end_x - view_m.start_x;
        int dy = view_m.end_y - view_m.start_y;
        
        // Match the 64-action encoding (y*16 + x*4 + dir)
        if (std::abs(dx) + std::abs(dy) == 1) {
            int d = (dy == -1) ? 0 : (dy == 1 ? 1 : (dx == -1 ? 2 : 3));
            int idx = (view_m.start_y * 4 + view_m.start_x) * 4 + d;
            accessor[idx] = 1.0f;
        }
    }
    return mask;
}
