#include "batched_stratego_env.h"

BatchedStrategoEnv::BatchedStrategoEnv(
    const stratego::BoardConfig& config,
    stratego::state::SetupType setup_type,
    int batch_size,
    int max_moves)
    : batch_size(batch_size)
{
    envs.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i)
        envs.push_back(std::make_unique<StrategoEnvironment>(config, setup_type, max_moves));

    int obs_channels = get_encoding_channels(config);
    int action_dim   = (int)envs[0]->action_dim();
    int H = config.height;
    int W = config.width;

    auto buf_opts = torch::TensorOptions().dtype(torch::kFloat32)
                        .pinned_memory(torch::cuda::is_available());
    obs_buf  = torch::zeros({batch_size, obs_channels, H, W}, buf_opts);
    mask_buf = torch::zeros({batch_size, action_dim},          buf_opts);
}

int BatchedStrategoEnv::get_action_channels() const {
    return envs[0]->get_action_encoder().get_action_channels();
}

std::pair<torch::Tensor, torch::Tensor> BatchedStrategoEnv::reset_all() {
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < batch_size; ++e) {
        obs_buf[e].copy_(envs[e]->reset());
        mask_buf[e].copy_(envs[e]->get_action_mask());
    }
    return {obs_buf, mask_buf};
}

BatchedStepResult BatchedStrategoEnv::step(const torch::Tensor& actions) {
    BatchedStepResult results;
    results.rewards.resize(batch_size, 0.0f);
    results.dones.resize(batch_size, false);

    auto actions_acc = actions.accessor<int64_t, 1>();

    #pragma omp parallel for schedule(static)
    for (int e = 0; e < batch_size; ++e) {
        auto step_res = envs[e]->step((int)actions_acc[e]);

        if (step_res.terminated) {
            results.rewards[e] = step_res.reward;
            results.dones[e]   = true;
            // Immediately auto-reset so callers always get a usable next obs/mask
            obs_buf[e].copy_(envs[e]->reset());
        } else {
            obs_buf[e].copy_(step_res.observation);
        }
        mask_buf[e].copy_(envs[e]->get_action_mask());
    }

    results.observations = obs_buf;
    results.masks        = mask_buf;
    return results;
}

std::vector<stratego::Player> BatchedStrategoEnv::get_current_players() const {
    std::vector<stratego::Player> players(batch_size);
    for (int e = 0; e < batch_size; ++e)
        players[e] = envs[e]->get_current_player();
    return players;
}
