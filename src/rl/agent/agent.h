#pragma once
// We didn't study PPO as much in class, so some of this is less clear to me

#include <vector>
#include <torch/torch.h>

// PPO stores past states and actions, then re-evaluates them with the updated
// policy to see how the model's behavior has changed. So, it is helpful to
// encapsulate the agent's output (action, log probability, and value) in a
// struct that can be easily stored and accessed during the PPO update phase
template <typename ActionType>
struct AgentOutput {
    ActionType action;  // Action to take in the environment
    float log_prob;     // Log probability that our model assigned to this action
    float value;        // Estimated value of the current state (for the critic)
    std::vector<float> mask; // Action mask used for this step
};

// Forward declaration of RolloutBuffer so we can pass it into the update function
template <typename ObsType, typename ActionType> class RolloutBuffer;

template <typename ObsType, typename ActionType>
class Agent {
public:
    virtual ~Agent() = default;

    // Used during the rollout phase (interacts with the Env)
    // Runs both actor and critic heads simultaneously for efficiency.
    [[nodiscard]] virtual AgentOutput<ActionType> act(const ObsType& obs, const torch::Tensor& mask) = 0;

    // Trigger neural net update
    virtual void update_weights(RolloutBuffer<ObsType, ActionType>& buffer) = 0;
};