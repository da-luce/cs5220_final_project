#pragma once
#include "stratego/policy.h"
#include "networks/model.h"
#include "rl/encoding/actions.h"
#include <torch/torch.h>
#include <memory>

class NeuralPolicy : public stratego::Policy {
public:
    NeuralPolicy(const std::string& model_path, const stratego::BoardConfig& config);
    NeuralPolicy(networks::StrategoNet model, const stratego::BoardConfig& config);

    stratego::Move get_move(const stratego::GameState& masked_state) override;
    bool is_human() const override { return false; }

private:
    networks::StrategoNet model{nullptr};
    stratego::BoardConfig config;
    encoding::actions::ActionEncoder action_encoder;

    torch::Tensor get_action_mask(const stratego::GameState& state);
};
