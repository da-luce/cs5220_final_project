#pragma once

#include "environment.h"
#include "stratego_env.h" // Your existing single environment
#include <vector>
#include <memory>
#include <torch/torch.h>

struct BatchedStepResult {
    torch::Tensor observations; // Shape: [batch_size, C, H, W]
    torch::Tensor masks;        // Shape: [batch_size, action_dim]
    std::vector<float> rewards; // Size: batch_size
    std::vector<bool> dones;    // Size: batch_size
};

class BatchedStrategoEnv {
public:
    BatchedStrategoEnv(const stratego::BoardConfig& config, 
                       stratego::state::SetupType setup_type, 
                       int batch_size, 
                       int max_moves = 60);

    // Returns a tuple of [Observations, Action Masks]
    std::pair<torch::Tensor, torch::Tensor> reset_all();

    // Takes a tensor of actions [batch_size] and steps all environments in parallel
    BatchedStepResult step(const torch::Tensor& actions);

    int get_batch_size() const { return batch_size; }
    int get_action_channels() const;
    std::vector<stratego::Player> get_current_players() const;

private:
    int batch_size;
    std::vector<std::unique_ptr<StrategoEnvironment>> envs;

    // Pre-allocated pinned memory buffers for fast GPU transfer
    torch::Tensor obs_buf;
    torch::Tensor mask_buf;
};