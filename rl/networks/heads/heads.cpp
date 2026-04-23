#include "networks/heads/heads.h"

namespace networks {
namespace heads {

PolicyHeadImpl::PolicyHeadImpl(int in_features, int out_actions) {
    fc = register_module("fc", torch::nn::Linear(in_features, out_actions));
}

torch::Tensor PolicyHeadImpl::forward(torch::Tensor x) {
    // Return raw logits (PPO update will use log_softmax)
    return fc->forward(x);
}

ValueHeadImpl::ValueHeadImpl(int in_features) {
    fc = register_module("fc", torch::nn::Linear(in_features, 1));
}

torch::Tensor ValueHeadImpl::forward(torch::Tensor x) {
    // Re-added tanh to match Python parity and ensure stable advantages
    return torch::tanh(fc->forward(x));
}

} // namespace heads
} // namespace networks
