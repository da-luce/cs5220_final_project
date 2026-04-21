#pragma once
#include "environment.h"
#include "../stratego/stratego.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// Our observation is a flattened vector representing the board state. This way
// we can feed this directly into your PyTorch/LibTorch or custom C++ CNN. The
// model will re-construct the tensor shape (C, H, W) from this flattened blob.
using StrategoObs = std::vector<float>;

// Instead of encoding actions as (start_x, start_y, dx, dy), meaning move piece
// at (start_x, start_y) by (dx, dy), we can flatten the action space by encoding
// each possible move as a unique integer.
// We do this because it simplifies the action output of the neural network to 
// a single scalar value. This is a standard RL technique.
using StrategoAction = int;

class StrategoEnvironment : public Environment<StrategoObs, StrategoAction> {
private:
    stratego::Board board;

    // Number of piece types that can be placed (excluding Empty and Water)
    int num_playable_piece_types;

    // Maximum move distance across the board
    [[nodiscard]] int get_max_dist() const {
        return std::max(board.get_width(), board.get_height()) - 1;
    }

public:

    explicit StrategoEnvironment(const stratego::Board& target_board)
        : board(target_board) {
        initialize_piece_channel_mapping();
    }

    // Helper to initialize piece_type_to_channel_offset
    std::map<stratego::PieceType, int> piece_type_to_channel_offset;
    void initialize_piece_channel_mapping() {
        piece_type_to_channel_offset.clear();
        std::vector<stratego::PieceType> sorted_piece_types;
        for (const auto& pair : board.get_config().piece_counts) {
            sorted_piece_types.push_back(pair.first);
        }
        // Sort to ensure consistent channel mapping across runs/platforms
        std::sort(sorted_piece_types.begin(), sorted_piece_types.end());

        num_playable_piece_types = sorted_piece_types.size();
        for (int i = 0; i < num_playable_piece_types; ++i) {
            piece_type_to_channel_offset[sorted_piece_types[i]] = i;
        }
    }

    void set_board(const stratego::Board& b) { board = b; }

    // Helper to calculate 1D CHW index
    [[nodiscard]] inline int get_chw_index(int c, int y, int x) const {
        return (c * board.get_height() * board.get_width()) + (y * board.get_width()) + x;
    }

    [[nodiscard]] StrategoObs encode_board(const stratego::Board& target_board) const {
        int w = target_board.get_width();
        int h = target_board.get_height();

        // We create a multi-channel binary representation of the board:
        // For each piece type that can be placed, we have:
        // - One channel for our pieces of that type
        // - One channel for revealed opponent pieces of that type
        // Then we have 3 additional channels:
        // - One channel for unrevealed opponent pieces (all types combined)
        // - One channel for water
        // - One channel for empty squares
        int total_channels = (2 * num_playable_piece_types) + 3;
        StrategoObs obs(total_channels * w * h, 0.0f); 
        stratego::Player me = target_board.get_current_turn();

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                stratego::Piece p = target_board.get_piece(x, y);

                int channel = -1;
                if (p.is_obstacle()) {
                    channel = 2 * num_playable_piece_types + 1; // Water channel
                } else if (p.is_empty()) {
                    channel = 2 * num_playable_piece_types + 2; // Empty channel
                } else {
                    if (p.owner == me) {
                        channel = piece_type_to_channel_offset.at(p.type); // Channels 0 to N-1
                    } else if (p.revealed) {
                        channel = num_playable_piece_types + 1 + piece_type_to_channel_offset.at(p.type); // Channels N+1 to 2N
                    } else {
                        channel = num_playable_piece_types; // Channel N for unrevealed opponent pieces
                    }
                }
                
                if (channel != -1) {

                    // IMPORTANT: If we're the opponent, we need to flip the board so that
                    // our pieces are always viewed from the same perspective.
                    // This way the NN doesn't have to learn two separate
                    // representations for playing as P0 vs P1.
                    int view_x = x;
                    int view_y = y;

                    if (target_board.get_current_turn() != stratego::Player::Red) {
                        view_x = w - 1 - x;
                        view_y = h - 1 - y;
                    }


                    // CHW Indexing
                    obs[get_chw_index(channel, view_y, view_x)] = 1.0f;
                }
            }
        }
        return obs;
    }

    [[nodiscard]] StrategoObs reset() override {
        board.initialize_game(stratego::SetupType::Random);
        return encode_board(board);
    }

    [[nodiscard]] StrategoAction encode_action(const stratego::Move& m) const {
        int w = board.get_width();
        int h = board.get_height();
        int max_dist = get_max_dist();

        // 1. Perspective Flip
        int vx = (board.get_current_turn() == stratego::Player::Red) ? m.start_x : (w - 1 - m.start_x);
        int vy = (board.get_current_turn() == stratego::Player::Red) ? m.start_y : (h - 1 - m.start_y);

        // 2. Relative Direction
        int dx = m.end_x - m.start_x;
        int dy = m.end_y - m.start_y;
        if (board.get_current_turn() != stratego::Player::Red) {
            dx = -dx;
            dy = -dy;
        }

        int dir = 0;
        if (dy < 0) dir = 0;      // Relative Up
        else if (dy > 0) dir = 1; // Relative Down
        else if (dx < 0) dir = 2; // Relative Left
        else if (dx > 0) dir = 3; // Relative Right

        int dist = std::max(std::abs(dx), std::abs(dy));

        return (vy * w + vx) * (4 * max_dist) + dir * max_dist + (dist - 1);
    }

    [[nodiscard]] stratego::Move decode_action(StrategoAction action) const {
        int w = board.get_width();
        int h = board.get_height();
        int max_dist = get_max_dist();

        // 1. Extract view-relative components
        int dist = (action % max_dist) + 1;
        int dir = (action / max_dist) % 4;
        int vx = (action / (max_dist * 4)) % w;
        int vy = (action / (max_dist * 4 * w));

        // 2. Map relative direction to relative displacement
        int rdx = 0, rdy = 0;
        if (dir == 0)      rdy = -dist; // Relative Up
        else if (dir == 1) rdy =  dist; // Relative Down
        else if (dir == 2) rdx = -dist; // Relative Left
        else if (dir == 3) rdx =  dist; // Relative Right

        // 3. Transform back to global board coordinates
        int start_x, start_y, end_x, end_y;

        if (board.get_current_turn() == stratego::Player::Red) {
            // Red's view is the global view
            start_x = vx;
            start_y = vy;
            end_x = start_x + rdx;
            end_y = start_y + rdy;
        } else {
            // Flip everything back for Blue
            start_x = w - 1 - vx;
            start_y = h - 1 - vy;
            // A relative "Up" for Blue is a global "Down" (+Y)
            end_x = start_x - rdx; 
            end_y = start_y - rdy;
        }

        return {start_x, start_y, end_x, end_y};
    }

    [[nodiscard]] StepResult<StrategoObs> step(const StrategoAction& action) override {
        stratego::Move move = decode_action(action);

        if (!board.is_legal_move(move)) {
            // Agent made an illegal move; penalize heavily and terminate
            return {encode_board(board), -1.0f, true, false};
        }

        stratego::CombatResult result = board.execute_move(move);

        float reward = -0.01f; // Non-winning moves have a small negative reward to encourage shorter games
        bool terminated = false;
        bool truncated = false;

        if (result == stratego::CombatResult::FlagCaptured) {
            reward = 1.0f; // Acting player captured the flag
            terminated = true;
        } else if (result == stratego::CombatResult::Draw) {
            truncated = true; // Typically triggered by max move limit being reached
        } else {
            // If the next player has no legal moves left, the acting player wins!
            auto next_moves = board.get_all_legal_moves(board.get_current_turn());
            if (next_moves.empty()) {
                reward = 1.0f;
                terminated = true;
            }
        }

        return {encode_board(board), reward, terminated, truncated};
    }

    [[nodiscard]] size_t observation_dim() const override {
        int w = board.get_width();
        int h = board.get_height();
        int total_channels = (2 * num_playable_piece_types) + 3;
        return total_channels * w * h;
    }

    [[nodiscard]] size_t action_dim() const override {
        // total cells * 4 directions * max_distance_per_direction
        return board.get_width() * board.get_height() * 4 * get_max_dist();
    }

    // Allows the Agent to mask invalid logit action branches before applying softmax
    [[nodiscard]] std::vector<StrategoAction> get_legal_actions() const {
        std::vector<StrategoAction> valid_actions;
        auto legal_moves = board.get_all_legal_moves(board.get_current_turn());
        
        for (const auto& m : legal_moves) {
            valid_actions.push_back(encode_action(m));
        }
        return valid_actions;
    }
    
    [[nodiscard]] const stratego::Board& get_board() const { 
        return board; 
    }
};
