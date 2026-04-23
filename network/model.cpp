#include "architectures/model.h"

namespace architectures {

StrategoNetImpl::StrategoNetImpl(std::shared_ptr<torsos::TorsoBase> torso_ptr, int D, int board_size) 
    : torso(torso_ptr) {
    
    // 1. Register the torso. 
    // This is critical! If you don't register it, the optimizer won't update its weights.
    register_module("torso", torso);

    // 2. Get the output depth from the torso dynamically
    int hidden_channels = torso->get_output_channels();

    // 3. Initialize and register the heads
    policy_head = register_module("policy_head", heads::PolicyHead(hidden_channels, D));
    value_head  = register_module("value_head",  heads::ValueHead(hidden_channels, board_size));
}

std::tuple<torch::Tensor, torch::Tensor> StrategoNetImpl::forward(torch::Tensor x) {
    // 1. Extract spatial features using the torso
    // Input: [Batch, In_Channels, H, W] -> Output: [Batch, Hidden_Channels, H, W]
    torch::Tensor features = torso->forward(x);

    // 2. Pass the extracted features to both heads independently
    torch::Tensor policy_logits = policy_head->forward(features); // [Batch, D, H, W]
    torch::Tensor value         = value_head->forward(features);  // [Batch, 1]

    // 3. Return both outputs
    return std::make_tuple(policy_logits, value);
}

} // namespace architectures