#include <vector>

// Notice that "observation" and "state" are used to represent the same idea,
// but "observation" is more general and can include partial information about 
// the state (since the agent may not have access to the full state of the
// environment).

/* Stores the result of taking a step/action in the environment
 * (s, a) -> (s', r, terminated, truncated)
 *           \                            /
 *            \  stores this information /
 */
template <typename ObsType>
struct StepResult {
    ObsType observation; // exact same as the next state, s'
    float reward;        // reward received after taking the action, r
    bool terminated;     // reached a terminal state (e.g., game over)
    bool truncated;      // reached time limit, but not a terminal state
};

template <typename ObsType, typename ActionType>
class Environment {
public:
    virtual ~Environment() = default;

    // Resets the environment to an initial state
    [[nodiscard]] virtual ObsType reset() = 0;

    // Takes a step in the environment
    [[nodiscard]] virtual StepResult<ObsType> step(const ActionType& action) = 0;

    // Optional but highly recommended for allocating buffers
    [[nodiscard]] virtual size_t observation_dim() const = 0;
    [[nodiscard]] virtual size_t action_dim() const = 0; 
};