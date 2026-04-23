#pragma once

#include <torch/torch.h>

namespace architectures {
namespace heads {

// ==========================================
// 1. Policy (Action) Head
// ==========================================
// Maps the torso output to a [Batch, D, H, W] tensor of action logits.
struct PolicyHeadImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};

    // in_channels: Output channels from your Torso (e.g., 128)
    // D: Maximum number of move types (e.g., 4 directions * 9 max distance = 36)
    PolicyHeadImpl(int in_channels, int D);

    // Returns raw logits. We return logits instead of Softmax because you 
    // will likely need to apply a "Legal Move Mask" (setting illegal moves to -infinity) 
    // before applying the final Softmax in your loss function or tree search.
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PolicyHead);

// ==========================================
// 2. Value Head
// ==========================================
// Evaluates the board state and returns a scalar between -1 (Loss) and 1 (Win).
struct ValueHeadImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    // in_channels: Output channels from your Torso (e.g., 128)
    // board_size: H * W (e.g., 100 for a 10x10 Stratego board)
    ValueHeadImpl(int in_channels, int board_size);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ValueHead);

} // namespace heads
} // namespace architectures