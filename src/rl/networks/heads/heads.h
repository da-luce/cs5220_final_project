#pragma once

#include <torch/torch.h>

namespace networks {
namespace heads {

// Policy head matching Python's simple Linear approach
struct PolicyHeadImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    PolicyHeadImpl() = default;
    PolicyHeadImpl(int in_features, int out_actions);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PolicyHead);

struct ValueHeadImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    ValueHeadImpl() = default;
    ValueHeadImpl(int in_features);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ValueHead);

} // namespace heads
} // namespace networks
