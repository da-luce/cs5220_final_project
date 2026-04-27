#pragma once

#include "agent.h"
#include "networks/model.h"
#include "rollout_buffer.h"
#include <torch/torch.h>

class PPOAgent : public Agent<torch::Tensor, int> {
public:
    PPOAgent(networks::StrategoNet model, 
             double lr, 
             double gamma, 
             double k_epochs, 
             double eps_clip);

    AgentOutput<int> act(const torch::Tensor& obs, const torch::Tensor& mask) override;
    void update_weights(RolloutBuffer<torch::Tensor, int>& buffer) override;

private:
    networks::StrategoNet model{nullptr};
    torch::optim::Adam optimizer;
    
    double gamma;
    double k_epochs;
    double eps_clip;

    torch::Tensor select_action(const torch::Tensor& logits, const torch::Tensor& mask);
};
