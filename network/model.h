#pragma once

#include <torch/torch.h>
#include <tuple>
#include <memory>

#include "architectures/torsos/cnn.h"
#include "architectures/heads/heads.h"

namespace architectures {

// ==========================================
// Full AlphaZero-style Network
// ==========================================
struct StrategoNetImpl : torch::nn::Module {
    // We use a shared_ptr to the base class so we can swap torsos easily
    std::shared_ptr<torsos::TorsoBase> torso;
    heads::PolicyHead policy_head{nullptr};
    heads::ValueHead value_head{nullptr};

    // Constructor parameters:
    // - torso_ptr: A pre-instantiated torso (e.g., CNNTorso)
    // - D: The maximum number of move distances/directions (for the policy head)
    // - board_size: H * W (for the value head's dense layer)
    StrategoNetImpl(std::shared_ptr<torsos::TorsoBase> torso_ptr, int D, int board_size);

    // Forward pass returns a tuple: {Policy Logits, Value}
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};

// Generates the shared pointer wrapper (StrategoNet)
TORCH_MODULE(StrategoNet);

} // namespace architectures