// Need to study rollout more
#pragma once

#include <vector>

template <typename ObsType, typename ActionType>
class RolloutBuffer {
public:
    std::vector<ObsType> observations;
    std::vector<ActionType> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> log_probs;
    std::vector<bool> is_terminals; // True if terminated OR truncated
    std::vector<std::vector<float>> masks; // Legal action masks
    
    // Populated post-rollout
    std::vector<float> advantages;
    std::vector<float> returns;

    void clear() {
        observations.clear();
        actions.clear();
        rewards.clear();
        values.clear();
        log_probs.clear();
        is_terminals.clear();
        masks.clear();
        advantages.clear();
        returns.clear();
    }

    void add(const ObsType& obs, const ActionType& action, float reward, 
             float value, float log_prob, bool is_terminal, const std::vector<float>& mask) {
        observations.push_back(obs);
        actions.push_back(action);
        rewards.push_back(reward);
        values.push_back(value);
        log_probs.push_back(log_prob);
        is_terminals.push_back(is_terminal);
        masks.push_back(mask);
    }
    
    // Generalized Advantage Estimation (GAE)
    // Call this right before passing the buffer to the Agent for updates
    void compute_advantages(float next_value, float gamma = 0.99f, float gae_lambda = 0.95f) {
        size_t step_count = rewards.size();
        advantages.assign(step_count, 0.0f);
        returns.assign(step_count, 0.0f);

        float last_gae_lam = 0.0f;
        
        // Calculate advantages backwards through the trajectory
        for (int step = step_count - 1; step >= 0; --step) {
            float next_non_terminal = 1.0f - static_cast<float>(is_terminals[step]);
            float next_val = (step == step_count - 1) ? next_value : values[step + 1];
            
            float delta = rewards[step] + gamma * next_val * next_non_terminal - values[step];
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam;
            
            advantages[step] = last_gae_lam;
            returns[step] = advantages[step] + values[step];
        }
    }
};