#pragma once
#include "environment.h"
#include "../stratego/stratego.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// Our observation is a flattened vector representing the board state. This way
// we can feed this directly into your PyTorch/LibTorch or custom C++ CNN
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

    // Helper: maximum move distance across the board
    [[nodiscard]] int get_max_dist() const {
        return std::max(board.get_width(), board.get_height()) - 1;
    }

public:
    void set_board(const stratego::Board& b) { board = b; }

    [[nodiscard]] StrategoObs encode_board(const stratego::Board& target_board) const {
        StrategoObs obs;
        int w = target_board.get_width();
        int h = target_board.get_height();

        // Our board has 9 channels
        // Channels 0-3: My pieces (Flag, Lieutenant, Captain, Major)
        // Channel 4: Unrevealed enemy pieces
        // Channels 5-8: Revealed enemy pieces (Flag, Lieutenant, Captain, Major)
        obs.reserve(w * h * 9);

        stratego::Player me = target_board.get_current_turn();

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                stratego::Piece p = target_board.get_piece(x, y);
                std::vector<float> cell(9, 0.0f);

                if (!p.is_empty() && !p.is_obstacle()) {
                    int idx = -1;
                    switch (p.type) {
                        case stratego::PieceType::Flag:       idx = 0; break;
                        case stratego::PieceType::Lieutenant: idx = 1; break;
                        case stratego::PieceType::Captain:    idx = 2; break;
                        case stratego::PieceType::Major:      idx = 3; break;
                        default: break;
                    }

                    if (idx != -1) {
                        if (p.owner == me) {
                            cell[idx] = 1.0f;
                        } else {
                            if (p.revealed) {
                                cell[idx + 5] = 1.0f;
                            } else {
                                cell[4] = 1.0f;
                            }
                        }
                    }
                }
                obs.insert(obs.end(), cell.begin(), cell.end());
            }
        }
        return obs;
    }

    StrategoEnvironment() {
        board.initialize_game(stratego::GameType::Tiny, stratego::SetupType::Random);
    }

    [[nodiscard]] StrategoObs reset() override {
        board.initialize_game(stratego::GameType::Tiny, stratego::SetupType::Random);
        return encode_board(board);
    }

    [[nodiscard]] StrategoAction encode_action(const stratego::Move& m) const {
        int w = board.get_width();
        int max_dist = get_max_dist();
        
        int dx = m.end_x - m.start_x;
        int dy = m.end_y - m.start_y;
        int dist = std::max(std::abs(dx), std::abs(dy));
        
        int dir = 0;
        if (dy < 0) dir = 0;
        else if (dy > 0) dir = 1;
        else if (dx < 0) dir = 2;
        else if (dx > 0) dir = 3;
        
        return (m.start_y * w + m.start_x) * (4 * max_dist) + dir * max_dist + (dist - 1);
    }

    [[nodiscard]] stratego::Move decode_action(StrategoAction action) const {
        int w = board.get_width();
        int max_dist = get_max_dist();

        int dist = (action % max_dist) + 1;
        int dir = (action / max_dist) % 4;
        int start_x = (action / (max_dist * 4)) % w;
        int start_y = (action / (max_dist * 4 * w));

        int dx = 0, dy = 0;
        if (dir == 0) dy = -dist;      // Up
        else if (dir == 1) dy = dist;  // Down
        else if (dir == 2) dx = -dist; // Left
        else if (dir == 3) dx = dist;  // Right

        return {start_x, start_y, start_x + dx, start_y + dy};
    }

    [[nodiscard]] StepResult<StrategoObs> step(const StrategoAction& action) override {
        stratego::Move move = decode_action(action);

        if (!board.is_legal_move(move)) {
            // Agent made an illegal move; penalize heavily and terminate
            return {encode_board(board), -1.0f, true, false};
        }

        stratego::CombatResult result = board.execute_move(move);

        float reward = 0.0f;
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
        return board.get_width() * board.get_height() * 9;
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