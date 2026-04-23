#pragma once

#include <torch/torch.h>
#include "base.h"

namespace architectures {
namespace torsos {

// Residual Block
struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};

    ResidualBlockImpl(int channels);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResidualBlock);

// 3. CNN Torso (Inherits from TorsoBase)
struct CNNTorsoImpl : TorsoBase {
    torch::nn::Sequential layers;
    int hidden_channels;

    // in_channels = (2N + 2) for your Stratego encoding
    // num_filters = hidden dimension (e.g., 64, 128, 256)
    // num_blocks  = number of residual blocks
    CNNTorsoImpl(int in_channels, int num_filters, int num_blocks);

    torch::Tensor forward(torch::Tensor x) override;
    int get_output_channels() const override;
};

TORCH_MODULE(CNNTorso);

} // namespace torsos
} // namespace architectures