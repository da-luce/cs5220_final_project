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
    auto features = get_board_encoding(masked_state, masked_state.current_turn);
    int C = get_encoding_channels(config);
    int H = config.height;
    int W = config.width;
    
    auto obs = torch::from_blob(features.data(), {1, C, H, W}, torch::kFloat32).clone();

    // 2. Forward pass
    auto [logits, value] = model->forward(obs);
    
    // 3. Apply mask in VIEW SPACE
    torch::Tensor mask = get_action_mask(masked_state);
    logits.view({1, -1}).masked_fill_(mask.unsqueeze(0) == 0, -1e9);

    // 4. Select best action index
    int action_idx = torch::argmax(logits.view({1, -1}), 1).item<int>();
    
    // 5. Decode action index back to global move
    int total_channels = action_encoder.get_action_channels();
    int channel = action_idx / (W * H);
    int spatial_idx = action_idx % (W * H);
    int vx = spatial_idx % W;
    int vy = spatial_idx / W;

    stratego::Move view_move = action_encoder.channel_to_move(channel, vx, vy);
    return (masked_state.current_turn == stratego::Player::Red) ? 
            view_move : stratego::Engine::get_flipped_move(config, view_move);
}

torch::Tensor NeuralPolicy::get_action_mask(const stratego::GameState& state) {
    int action_dim = action_encoder.get_action_channels() * config.width * config.height;
    auto mask = torch::zeros({action_dim}, torch::kFloat32);
    auto accessor = mask.accessor<float, 1>();
    auto legal_moves = stratego::Engine::get_all_legal_moves(state, state.current_turn);
    
    int spatial_size = config.width * config.height;

    for (const auto& m : legal_moves) {
        stratego::Move view_m = (state.current_turn == stratego::Player::Red) ? 
                                 m : stratego::Engine::get_flipped_move(config, m);
        try {
            int channel = action_encoder.move_to_channel(view_m);
            int spatial_idx = (view_m.start_y * config.width) + view_m.start_x;
            int idx = (channel * spatial_size) + spatial_idx;
            accessor[idx] = 1.0f;
        } catch (...) {
            continue;
        }
    }
    return mask;
}
