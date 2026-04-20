// We didn't study PPO as much in class, so some of this is less clear to me

// PPO stores past states and actions, then re-evaluates them with the updated
// policy to see how the model's behavior has changed. So, it is helpful to
// encapsulate the agent's output (action, log probability, and value) in a
// struct that can be easily stored and accessed during the PPO update phase
template <typename ActionType>
struct AgentOutput {
    ActionType action;  // Action to take in the environment
    float log_prob;     // Log probability that our model assigned to this action
    float value;        // Estimated value of the current state (for the critic)
};

template <typename ObsType, typename ActionType>
class Agent {
public:
    virtual ~Agent() = default;

    // Used during the rollout phase (interacts with the Env)
    // Runs both actor and critic heads simultaneously for efficiency.
    [[nodiscard]] virtual AgentOutput<ActionType> act(const ObsType& obs) = 0;

    // Used during the PPO update phase.
    // Given a batch of states and previously taken actions, return new log_probs, 
    // values, and entropy (for exploration bonus)
    virtual void evaluate_actions(
        const std::vector<ObsType>& obs_batch,
        const std::vector<ActionType>& action_batch,
        std::vector<float>& out_log_probs,
        std::vector<float>& out_values,
        std::vector<float>& out_entropy
    ) = 0;
    
    // Trigger the actual neural net update
    virtual void update_weights() = 0;
};