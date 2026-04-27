#pragma once

#include "agent.h"
#include "networks/model.h"
#include "rollout_buffer.h"
#include <torch/torch.h>
#include <vector>

struct BatchedAgentOutput {
    std::vector<int> actions;
    std::vector<float> log_probs;
    std::vector<float> values;
};

struct PPOStats {
    float policy_loss   = 0.0f;
    float value_loss    = 0.0f;
    float entropy       = 0.0f;
    float kl_divergence = 0.0f;
};

class PPOAgent : public Agent<torch::Tensor, int> {
public:
    PPOAgent(networks::StrategoNet model,
             double lr,
             double gamma,
             double k_epochs,
             double eps_clip);

    AgentOutput<int> act(const torch::Tensor& obs, const torch::Tensor& mask) override;
    BatchedAgentOutput act_batch(const torch::Tensor& obs_batch, const torch::Tensor& mask_batch);
    void update_weights(RolloutBuffer<torch::Tensor, int>& buffer) override;

    PPOStats last_stats() const { return last_stats_; }

private:
    networks::StrategoNet model{nullptr};
    torch::optim::Adam optimizer;

    double gamma;
    double k_epochs;
    double eps_clip;

    PPOStats last_stats_;

    torch::Tensor select_action(const torch::Tensor& logits, const torch::Tensor& mask);
};
