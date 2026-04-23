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
    int total_channels = action_encoder.get_action_channels();
    int channel = action_idx / (config.width * config.height);
    int spatial_idx = action_idx % (config.width * config.height);
    int x = spatial_idx % config.width;
    int y = spatial_idx / config.width;

    stratego::Move view_move = action_encoder.channel_to_move(channel, x, y);
    
    // 3. Transform to Global Space
    stratego::Move global_move = (state.current_turn == stratego::Player::Red) ? 
                                 view_move : stratego::Engine::get_flipped_move(config, view_move);
    
    StepResult<torch::Tensor> result{};
    result.reward = 0.0f;
    result.terminated = false;
    result.truncated = false;
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
        // Always check for trapping to match Python behavior
        if (stratego::Engine::get_all_legal_moves(state, state.current_turn).empty()) {
            result.reward = 1.0f;
            result.terminated = true;
        }
    }

    result.observation = get_current_obs();
    return result;
}

torch::Tensor StrategoEnvironment::get_current_obs() const {
    auto features = get_board_encoding(state, state.current_turn);
    int C = get_encoding_channels(config);
    int H = config.height;
    int W = config.width;
    
    auto obs = torch::from_blob(features.data(), {C, H, W}, torch::kFloat32).clone();
    return obs;
}

size_t StrategoEnvironment::observation_dim() const { 
    return get_encoding_channels(config) * config.width * config.height; 
}

size_t StrategoEnvironment::action_dim() const { 
    return action_encoder.get_action_channels() * config.width * config.height; 
}

torch::Tensor StrategoEnvironment::get_action_mask() const {
    auto mask = torch::zeros({(long)action_dim()}, torch::kFloat32);
    auto accessor = mask.accessor<float, 1>();
    auto legal_moves = stratego::Engine::get_all_legal_moves(state, state.current_turn);
    
    int total_channels = action_encoder.get_action_channels();
    int spatial_size = config.width * config.height;

    for (const auto& m : legal_moves) {
        stratego::Move view_m = (state.current_turn == stratego::Player::Red) ? 
                                 m : stratego::Engine::get_flipped_move(config, m);
        
        try {
            int channel = action_encoder.move_to_channel(view_m);
            int spatial_idx = (view_m.start_y * config.width) + view_m.start_x;
            int action_idx = (channel * spatial_size) + spatial_idx;
            accessor[action_idx] = 1.0f;
        } catch (const std::exception& e) {
            // Should not happen for legal moves if encoder is correct
            continue;
        }
    }
    return mask;
}
