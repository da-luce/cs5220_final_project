// More work needed here
#pragma once
#include <functional>

template <typename ObsType, typename ActionType>
void train_ppo(
    Environment<ObsType, ActionType>& env, 
    Agent<ObsType, ActionType>& agent, 
    RolloutBuffer<ObsType, ActionType>& buffer,
    int total_timesteps, 
    int rollout_length,
    std::function<void(int)> on_rollout_end = nullptr
) {
    ObsType obs = env.reset();

    for (int t = 0; t < total_timesteps; t += rollout_length) {
        
        // 1. COLLECT TRAJECTORIES
        for (int step = 0; step < rollout_length; ++step) {
            // Forward pass (no gradients needed here)
            AgentOutput<ActionType> out = agent.act(obs);
            
            // Step environment
            StepResult<ObsType> next = env.step(out.action);
            bool is_terminal = next.terminated || next.truncated;

            // Store transition
            buffer.add(obs, out.action, next.reward, out.value, out.log_prob, is_terminal, out.mask);

            if (next.terminated) {
                obs = env.reset();
            } else {
                obs = next.observation;
            }
        }

        // 2. COMPUTE GAE
        // We need the value of the *next* state to bootstrap the final advantage
        float next_value = agent.act(obs).value; 
        buffer.compute_advantages(next_value, /*gamma=*/0.99f, /*gae_lambda=*/0.95f);

        // 3. UPDATE NETWORK
        // agent.update_weights will internally handle mini-batching and the PPO loss math
        agent.update_weights(buffer);

        // 4. RESET BUFFER
        buffer.clear();
        
        // 5. EVALUATION CALLBACK
        if (on_rollout_end) {
            on_rollout_end(t + rollout_length);
        }
    }
}