#pragma once
#include <vector>
#include <torch/torch.h>

template <typename ObsType, typename ActionType>
struct RolloutBuffer {
    std::vector<ObsType> observations;
    std::vector<ActionType> actions;
    std::vector<float> log_probs;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<bool> is_terminals;
    std::vector<torch::Tensor> masks;

    void clear() {
        observations.clear();
        actions.clear();
        log_probs.clear();
        rewards.clear();
        values.clear();
        is_terminals.clear();
        masks.clear();
    }

    size_t size() const {
        return observations.size();
    }
};
