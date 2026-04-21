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

    for (int t = 0; t < total_timesteps; ) {
        int steps_this_batch = 0;

        // Collect full episodes until we hit our batch size (rollout_length)
        while (steps_this_batch < rollout_length) {
            std::vector<int> episode_indices;
            bool game_over = false;

            // Run one full episode
            while (!game_over) {
                AgentOutput<ActionType> out = agent.act(obs);
                StepResult<ObsType> next = env.step(out.action);
                
                // Track where this transition is in the buffer
                episode_indices.push_back(buffer.observations.size());
                
                buffer.add(obs, out.action, next.reward, out.value, out.log_prob, 
                           next.terminated || next.truncated, out.mask);

                obs = next.observation;
                steps_this_batch++;
                game_over = next.terminated || next.truncated;

                if (game_over) {
                    // Python logic: Back-propagate the final reward to all moves in this game
                    float final_reward = next.reward; 
                    // Note: If your env returns reward only at the end, 
                    // you must distribute it to the steps in episode_indices.
                    
                    for (size_t i = 0; i < episode_indices.size(); ++i) {
                        int idx = episode_indices[i];
                        // Alternate rewards based on turn. The last player is the one who ended the game.
                        bool is_last_player = (i % 2) == ((episode_indices.size() - 1) % 2);
                        buffer.rewards[idx] = is_last_player ? final_reward : -final_reward;
                    }
                    
                    obs = env.reset();
                }
            }
        }

        // Python doesn't use GAE bootstrapping. It uses simple discounted returns.
        // You should update compute_advantages to calculate Monte Carlo returns:
        // Returns = Reward + gamma * Reward_next + ...
        buffer.compute_monte_carlo_returns(0.99f); 

        agent.update_weights(buffer);
        t += steps_this_batch;
        buffer.clear();
        if (on_rollout_end) on_rollout_end(t);
    }
}