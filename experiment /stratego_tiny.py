import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
import os

# ==========================================
# 1. ENVIRONMENT (GAME LOGIC)
# ==========================================
class StrategoTinyEnv:
    def __init__(self):
        self.board_size = 4
        self.max_turns = 60
        self.pieces = [0, 5, 6, 7] # Flag=0, Lieut=5, Capt=6, Major=7
        self.reset()

    def reset(self):
        self.turn_count = 0
        self.current_player = 0
        self.history = [] 
        
        self.board = np.full((self.board_size, self.board_size, 3), -1)
        
        p0_pieces = np.random.permutation(self.pieces)
        p1_pieces = np.random.permutation(self.pieces)
        
        for col in range(self.board_size):
            self.board[3, col] = [0, p0_pieces[col], 0] 
            self.board[0, col] = [1, p1_pieces[col], 0] 
            
        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros((9, self.board_size, self.board_size), dtype=np.float32)
        for r in range(self.board_size):
            for c in range(self.board_size):
                owner, rank, revealed = self.board[r, c]
                if owner == -1: continue
                
                view_r = r if self.current_player == 0 else 3 - r
                view_c = c if self.current_player == 0 else 3 - c
                
                if owner == self.current_player:
                    idx = self.pieces.index(rank)
                    obs[idx, view_r, view_c] = 1.0
                else:
                    if revealed == 1:
                        idx = self.pieces.index(rank) + 5
                        obs[idx, view_r, view_c] = 1.0
                    else:
                        obs[4, view_r, view_c] = 1.0 
        return obs

    def get_legal_actions(self):
        legal = np.zeros(64, dtype=np.float32)
        has_legal_move = False
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                owner, rank, _ = self.board[r, c]
                if owner == self.current_player and rank != 0: 
                    
                    dir_r = [-1, 1, 0, 0] if self.current_player == 0 else [1, -1, 0, 0]
                    dir_c = [0, 0, -1, 1] if self.current_player == 0 else [0, 0, 1, -1]
                    
                    for d in range(4):
                        tr, tc = r + dir_r[d], c + dir_c[d]
                        if 0 <= tr < self.board_size and 0 <= tc < self.board_size:
                            if self.board[tr, tc, 0] != self.current_player:
                                view_r = r if self.current_player == 0 else 3 - r
                                view_c = c if self.current_player == 0 else 3 - c
                                action_idx = view_r * 16 + view_c * 4 + d
                                legal[action_idx] = 1.0
                                has_legal_move = True
        return legal, has_legal_move

    def step(self, action):
        view_r = action // 16
        view_c = (action % 16) // 4
        d = action % 4
        
        r = view_r if self.current_player == 0 else 3 - view_r
        c = view_c if self.current_player == 0 else 3 - view_c
        
        dir_r = [-1, 1, 0, 0] if self.current_player == 0 else [1, -1, 0, 0]
        dir_c = [0, 0, -1, 1] if self.current_player == 0 else [0, 0, 1, -1]
        
        tr, tc = r + dir_r[d], c + dir_c[d]
        
        mover_owner, mover_rank, mover_rev = self.board[r, c]
        target_owner, target_rank, target_rev = self.board[tr, tc]
        
        self.board[r, c] = [-1, -1, 0] 
        done = False
        winner = -1 
        
        if target_owner == -1:
            self.board[tr, tc] = [mover_owner, mover_rank, mover_rev]
        else:
            if target_rank == 0: 
                self.board[tr, tc] = [mover_owner, mover_rank, 1]
                done = True
                winner = self.current_player
            elif mover_rank > target_rank:
                self.board[tr, tc] = [mover_owner, mover_rank, 1]
            elif mover_rank < target_rank:
                self.board[tr, tc] = [target_owner, target_rank, 1]
            else:
                self.board[tr, tc] = [-1, -1, 0]
                
        self.history.append(self.current_player)
        self.turn_count += 1
        self.current_player = 1 - self.current_player
        
        if not done:
            if self.turn_count >= self.max_turns:
                done = True
                winner = -1
            else:
                _, has_moves = self.get_legal_actions()
                if not has_moves:
                    done = True
                    winner = 1 - self.current_player 
        
        return self._get_observation(), done, winner

# ==========================================
# 2. NEURAL NETWORK
# ==========================================
class StrategoNet(nn.Module):
    def __init__(self):
        super(StrategoNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(256, 64)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x, legal_actions_mask):
        x = torch.FloatTensor(x)
        if len(x.shape) == 3: x = x.unsqueeze(0)
        
        features = self.shared_fc(self.conv(x))
        logits = self.actor_head(features)
        
        mask = torch.FloatTensor(legal_actions_mask)
        if len(mask.shape) == 1: mask = mask.unsqueeze(0)
        
        logits = logits + ((1.0 - mask) * -1e9)
        probs = torch.softmax(logits, dim=-1)
        value = torch.tanh(self.critic_head(features))
        
        return probs, value

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_vs_champion(challenger, champion, num_games=100):
    """Pits the current training model against the saved best model."""
    env = StrategoTinyEnv()
    challenger.eval()
    champion.eval()
    
    challenger_wins = 0
    draws = 0
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        
        # Swap who plays Player 0 to remove first-mover bias
        challenger_is_p0 = (game % 2 == 0)
        
        while not done:
            mask, has_legal_move = env.get_legal_actions()
            if not has_legal_move:
                done = True
                winner = 1 - env.current_player
                break
                
            # Determine whose turn it is
            if (env.current_player == 0 and challenger_is_p0) or \
               (env.current_player == 1 and not challenger_is_p0):
                active_net = challenger
            else:
                active_net = champion
                
            with torch.no_grad():
                probs, _ = active_net(obs, mask)
                # Ensure we only pick strictly legal moves during eval
                probs = probs * torch.FloatTensor(mask)
                if probs.sum() > 0:
                    action = torch.argmax(probs).item() # Deterministic play for eval
                else:
                    # Failsafe
                    legal_indices = np.where(mask == 1.0)[0]
                    action = np.random.choice(legal_indices)
                    
            obs, done, winner = env.step(action)
            
        if winner == -1:
            draws += 1
        elif (winner == 0 and challenger_is_p0) or (winner == 1 and not challenger_is_p0):
            challenger_wins += 1
            
    challenger.train() # Set back to training mode
    win_rate = challenger_wins / num_games
    draw_rate = draws / num_games
    return win_rate, draw_rate

# ==========================================
# 4. PPO TRAINING LOOP
# ==========================================
def train_self_play():
    env = StrategoTinyEnv()
    
    # Initialize Challenger (training) and Champion (frozen baseline)
    challenger_net = StrategoNet()
    champion_net = StrategoNet()
    champion_net.load_state_dict(challenger_net.state_dict())
    
    optimizer = optim.Adam(challenger_net.parameters(), lr=1e-4)
    
    epochs = 20000
    gamma = 0.99 
    eps_clip = 0.2
    
    eval_freq = 500       # Evaluate every X epochs
    eval_games = 100      # Games to play during evaluation
    win_threshold = 0.55  # Challenger must win > 55% to become the new Champion
    
    print("Starting Self-Play Training...")
    
    torch.save(champion_net.state_dict(), "stratego_random_weights.pth")
    
    
    for epoch in range(1, epochs + 1):
        obs = env.reset()
        done = False
        
        states, actions, log_probs, values, masks, players = [], [], [], [], [], []
        
        while not done:
            mask, has_legal_move = env.get_legal_actions()
            
            if not has_legal_move:
                done = True
                winner = 1 - env.current_player
                break
                
            probs, value = challenger_net(obs, mask)
            m = Categorical(probs)
            action = m.sample()
            
            states.append(obs)
            actions.append(action)
            log_probs.append(m.log_prob(action))
            values.append(value)
            masks.append(mask)
            players.append(env.current_player)
            
            obs, done, winner = env.step(action.item())

        # Reward Assignment
        step_rewards = []
        for p in players:
            if winner == -1:
                step_rewards.append(0.0)
            elif p == winner:
                step_rewards.append(1.0)
            else:
                step_rewards.append(-1.0)

        # Discounted Returns
        returns = []
        discounted_sum = 0
        for r in reversed(step_rewards):
            discounted_sum = r + (gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        old_log_probs = torch.stack(log_probs).detach().squeeze()
        values_tensor = torch.stack(values).squeeze().detach()
        
        if len(returns.shape) == 0: returns = returns.unsqueeze(0)
        if len(values_tensor.shape) == 0: values_tensor = values_tensor.unsqueeze(0)
        if len(old_log_probs.shape) == 0: old_log_probs = old_log_probs.unsqueeze(0)

        advantages = returns - values_tensor

        # PPO Update
        optimizer.zero_grad()
        new_probs, new_values = challenger_net(np.array(states), np.array(masks))
        new_m = Categorical(new_probs)
        new_log_probs = new_m.log_prob(torch.tensor(actions))
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus to encourage exploration
        entropy_loss = new_m.entropy().mean()
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
        loss.backward()
        optimizer.step()
        
        # --- PERIODIC EVALUATION ---
        if epoch % eval_freq == 0:
            print(f"\n--- Epoch {epoch}: Evaluating Challenger vs Champion ---")
            win_rate, draw_rate = evaluate_vs_champion(challenger_net, champion_net, num_games=eval_games)
            print(f"Challenger Win Rate: {win_rate*100:.1f}% | Draw Rate: {draw_rate*100:.1f}%")
            
            if win_rate >= win_threshold:
                print(">>> CHALLENGER IS THE NEW CHAMPION! Saving model... <<<")
                champion_net.load_state_dict(challenger_net.state_dict())
                torch.save(champion_net.state_dict(), "stratego_best_model.pth")
            else:
                print("Challenger failed to dethrone the Champion. Continuing training...")
            print("-" * 50 + "\n")

if __name__ == "__main__":
    train_self_play()