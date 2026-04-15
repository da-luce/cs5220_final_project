// Need to study rollout more

template <typename ObsType, typename ActionType>
class RolloutBuffer {
public:
    std::vector<ObsType> observations;
    std::vector<ActionType> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> log_probs;
    std::vector<bool> is_terminals; // True if terminated OR truncated
    
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
        advantages.clear();
        returns.clear();
    }

    void add(const ObsType& obs, const ActionType& action, float reward, 
             float value, float log_prob, bool is_terminal) {
        observations.push_back(obs);
        actions.push_back(action);
        rewards.push_back(reward);
        values.push_back(value);
        log_probs.push_back(log_prob);
        is_terminals.push_back(is_terminal);
    }
    
    // Generalized Advantage Estimation (GAE)
    // Call this right before passing the buffer to the Agent for updates
    void compute_advantages(float gamma = 0.99f, float gae_lambda = 0.95f) {
        // Implementation omitted for brevity
    }
};