import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import os

# IMPORTANT: Ensure this matches the network class you trained!
# If you used StrategoNetLarge in training, import StrategoNetLarge instead.
from stratego_tiny import StrategoTinyEnv, StrategoNet 

class StrategoGUI:
    def __init__(self, master, model_path="stratego_best_model.pth"):
        self.master = master
        master.title("Stratego Tiny vs AI")
        master.geometry("400x450")
        
        self.env = StrategoTinyEnv()
        self.net = StrategoNet()
        
        # Load the AI model
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, weights_only=True))
            self.net.eval()
            print("Model loaded successfully.")
        else:
            messagebox.showwarning("Model Missing", f"Could not find {model_path}. The AI will play randomly!")

        self.obs = self.env.reset()
        self.selected_sq = None  # To track which piece the player clicked first
        
        # --- UI Layout ---
        self.info_label = tk.Label(master, text="Your Turn! (Green)", font=("Arial", 14, "bold"))
        self.info_label.pack(pady=15)
        
        self.board_frame = tk.Frame(master)
        self.board_frame.pack()
        
        # Create a 4x4 grid of buttons
        self.buttons = [[None for _ in range(4)] for _ in range(4)]
        for r in range(4):
            for c in range(4):
                btn = tk.Button(
                    self.board_frame, text="", width=6, height=3,
                    font=("Arial", 16, "bold"),
                    command=lambda row=r, col=c: self.on_square_click(row, col)
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                self.buttons[r][c] = btn
                
        self.update_board_ui()

    def update_board_ui(self):
        """Updates the colors and text of the buttons based on the game state."""
        rank_to_char = {0: 'F', 5: 'L', 6: 'C', 7: 'M', -1: ''}
        
        for r in range(4):
            for c in range(4):
                owner, rank, revealed = self.env.board[r, c]
                btn = self.buttons[r][c]
                
                if owner == -1:
                    btn.config(text="", bg="#e0e0e0", state=tk.NORMAL) # Empty square
                elif owner == 0:
                    # Human piece
                    btn.config(text=rank_to_char[rank], bg="#90ee90", fg="black")
                elif owner == 1:
                    # AI piece
                    if revealed == 1:
                        btn.config(text=rank_to_char[rank], bg="#ffcccb", fg="darkred")
                    else:
                        btn.config(text="?", bg="#ff4c4c", fg="white")
                        
        # Highlight selected piece
        if self.selected_sq:
            r, c = self.selected_sq
            self.buttons[r][c].config(bg="#ffff99") # Yellow highlight

    def on_square_click(self, r, c):
        """Triggered when the user clicks any square on the board."""
        if self.env.current_player != 0:
            return  # Ignore clicks if it's the AI's turn
            
        owner, rank, _ = self.env.board[r, c]
        
        # Phase 1: Selecting a piece to move
        if self.selected_sq is None:
            if owner == 0 and rank != 0:  # Must be your piece, and not the Flag
                self.selected_sq = (r, c)
                self.update_board_ui()
            return
            
        # Phase 2: Deselecting if you click the same piece again
        if self.selected_sq == (r, c):
            self.selected_sq = None
            self.update_board_ui()
            return
            
        # Phase 3: Attempting to move to a destination
        start_r, start_c = self.selected_sq
        direction = self.get_direction(start_r, start_c, r, c)
        
        if direction == -1:
            self.info_label.config(text="Invalid move. Pick adjacent square.", fg="red")
            self.selected_sq = None
            self.update_board_ui()
            return
            
        # Convert to action index (Row * 16 + Col * 4 + Dir)
        action_idx = start_r * 16 + start_c * 4 + direction
        mask, has_moves = self.env.get_legal_actions()
        
        if not has_moves or mask[action_idx] == 0:
            self.info_label.config(text="Illegal move!", fg="red")
            self.selected_sq = None
            self.update_board_ui()
            return
            
        # -- Execute Human Move --
        self.obs, done, winner = self.env.step(action_idx)
        self.selected_sq = None
        self.update_board_ui()
        
        if done:
            self.handle_game_over(winner)
        else:
            self.info_label.config(text="AI is thinking...", fg="black")
            self.master.update()
            # Wait 600ms so it feels like the AI is thinking, then take AI turn
            self.master.after(600, self.ai_turn)

    def get_direction(self, sr, sc, tr, tc):
        """Converts start and target coordinates into a direction integer."""
        if tr == sr - 1 and tc == sc: return 0  # Up
        if tr == sr + 1 and tc == sc: return 1  # Down
        if tr == sr and tc == sc - 1: return 2  # Left
        if tr == sr and tc == sc + 1: return 3  # Right
        return -1 # Invalid diagonal or distant move

    def ai_turn(self):
        """Executes the AI's forward pass and updates the game."""
        mask, has_moves = self.env.get_legal_actions()
        if not has_moves:
            self.handle_game_over(0) # AI trapped, Human wins
            return
            
        with torch.no_grad():
            probs, _ = self.net(self.obs, mask)
            # Ensure illegal moves have 0 probability
            probs = probs * torch.FloatTensor(mask) 
            
            if probs.sum() > 0:
                action = torch.argmax(probs).item()
            else:
                # Failsafe fallback
                legal_indices = np.where(mask == 1.0)[0]
                action = np.random.choice(legal_indices)
                
        self.obs, done, winner = self.env.step(action)
        self.update_board_ui()
        
        if done:
            self.handle_game_over(winner)
        else:
            self.info_label.config(text="Your Turn! (Green)", fg="black")

    def handle_game_over(self, winner):
        """Ends the game and shows a popup box."""
        if winner == 0:
            msg = "YOU WIN! You beat the AI!"
            self.info_label.config(text=msg, fg="green")
        elif winner == 1:
            msg = "AI WINS! It outsmarted you."
            self.info_label.config(text=msg, fg="red")
        else:
            msg = "IT'S A TIE!"
            self.info_label.config(text=msg, fg="orange")
            
        # Disable all buttons so no more moves can be made
        for r in range(4):
            for c in range(4):
                self.buttons[r][c].config(state=tk.DISABLED)
                
        messagebox.showinfo("Game Over", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = StrategoGUI(root)
    root.mainloop()