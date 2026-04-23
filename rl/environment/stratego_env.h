#pragma once

#include "environment.h"
#include "stratego/state.h"
#include "stratego/engine.h"
#include "stratego/board.h"
#include "rl/encoding/board.h"
#include "rl/encoding/actions.h"
#include <torch/torch.h>
#include <memory>

class StrategoEnvironment : public Environment<torch::Tensor, int> {
public:
    StrategoEnvironment(stratego::BoardConfig config, 
                        stratego::state::SetupType setup_type,
                        int max_moves = 60);

    torch::Tensor reset() override;
    StepResult<torch::Tensor> step(const int& action_idx) override;

    size_t observation_dim() const override;
    size_t action_dim() const override;

    torch::Tensor get_action_mask() const;
    stratego::Player get_current_player() const { return state.current_turn; }

private:
    stratego::BoardConfig config;
    stratego::state::SetupType setup_type;
    int max_moves;

    stratego::GameState state;
    encoding::actions::ActionEncoder action_encoder;

    torch::Tensor get_current_obs() const;
};
