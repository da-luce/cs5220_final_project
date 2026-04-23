#include "networks/model.h"

namespace networks {

StrategoNetImpl::StrategoNetImpl(std::shared_ptr<torsos::TorsoBase> torso_ptr, int out_actions, int board_size) 
    : torso(torso_ptr) {
    
    register_module("torso", torso);

    int hidden_channels = torso->get_output_channels();
    int flattened_size = hidden_channels * board_size;

    shared_fc = register_module("shared_fc", torch::nn::Linear(flattened_size, 256));

    // Policy head maps directly to the flat action space (e.g. 64 for 4x4)
    policy_head = register_module("policy_head", heads::PolicyHead(256, out_actions));
    value_head  = register_module("value_head",  heads::ValueHead(256));
}

std::tuple<torch::Tensor, torch::Tensor> StrategoNetImpl::forward(torch::Tensor x) {
    torch::Tensor features = torso->forward(x);
    features = features.view({features.size(0), -1});
    features = torch::relu(shared_fc->forward(features));

    torch::Tensor policy_logits = policy_head->forward(features);
    torch::Tensor value         = value_head->forward(features);

    return std::make_tuple(policy_logits, value);
}

} // namespace networks
