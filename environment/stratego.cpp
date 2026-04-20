#include "environment.h"
#include "../stratego/stratego.h"
#include <vector>
#include <stdexcept>
#include <iostream>

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
    int board_width;
    int board_height;
    int max_steps;
    int current_step;

    stratego::Board board;

    // Encode and flatten the internal board state into an observation vector
    // for the agent.
    StrategoObs get_masked_observation() const {
        size_t obs_size = observation_dim();
        StrategoObs obs(obs_size, 0.0f);
        
        stratego::Player current_player = board.get_current_turn();
        
        for (int y = 0; y < board_height; ++y) {
            for (int x = 0; x < board_width; ++x) {
                stratego::Piece p = board.get_piece(x, y);
                int base_idx = (y * board_width + x) * 3;
                
                if (p.is_obstacle()) {
                    obs[base_idx + 2] = 1.0f; // Obstacle channel
                } else if (!p.is_empty()) {
                    if (p.owner == current_player) {
                        obs[base_idx + 0] = static_cast<float>(p.type); // Player piece channel
                    } else {
                        // Opponent piece channel: mask rank if unrevealed
                        obs[base_idx + 1] = p.revealed ? static_cast<float>(p.type) : -1.0f;
                    }
                }
            }
        }
        
        return obs;
    }

    void decode_action(StrategoAction action, int& start_x, int& start_y, int& dx, int& dy, int& distance) {
        // Stratego Constants
        const int MAX_DISTANCE = 9; // Max slide for a Scout on a 10x10 board
        const int ACTIONS_PER_SQUARE = 4 * MAX_DISTANCE; // 36 possible moves per square

        // 1. Isolate the starting square (Result: 0 to 99)
        int source_square = action / ACTIONS_PER_SQUARE;
        start_y = source_square / board_height; // Row
        start_x = source_square % board_width;  // Column

        // 2. Isolate the specific move from that square (Result: 0 to 35)
        int move_index = action % ACTIONS_PER_SQUARE;
        
        // 3. Extract direction and distance
        // Direction: 0=North, 1=South, 2=East, 3=West
        int direction = move_index / MAX_DISTANCE; 
        
        // Distance: 1 to 9 squares (adding 1 because a move of distance 0 is invalid)
        distance = (move_index % MAX_DISTANCE) + 1; 

        // 4. Map direction integer to a movement vector (dx, dy)
        dx = 0; 
        dy = 0;
        
        switch (direction) {
            case 0: dy = -1; break; // North (assuming 0,0 is top-left)
            case 1: dy =  1; break; // South
            case 2: dx =  1; break; // East
            case 3: dx = -1; break; // West
        }
    }

public:
    StrategoEnvironment(int w = 10, int h = 10, int max_s = 500) 
        : board_width(w), board_height(h), max_steps(max_s), current_step(0), board(w, h) {
    }

    // --- Core Interface Implementation ---

    [[nodiscard]] StrategoObs reset() override {
        current_step = 0;
        board.initialize_empty();
        
        // Initialize basic piece placement for testing
        // For self-play, you might want to randomize the initial setups
        // from a pool of valid configurations.
        board.place_piece(0, 0, stratego::PieceType::Flag, stratego::Player::Red);
        board.place_piece(1, 0, stratego::PieceType::Bomb, stratego::Player::Red);
        board.place_piece(2, 0, stratego::PieceType::Scout, stratego::Player::Red);
        
        board.place_piece(9, 9, stratego::PieceType::Flag, stratego::Player::Blue);
        board.place_piece(8, 9, stratego::PieceType::Bomb, stratego::Player::Blue);
        board.place_piece(7, 9, stratego::PieceType::Scout, stratego::Player::Blue);

        return get_masked_observation();
    }

    [[nodiscard]] StepResult<StrategoObs> step(const StrategoAction& action) override {
        current_step++;
        StepResult<StrategoObs> result;
        result.truncated = false;
        result.terminated = false;
        result.reward = 0.0f;

        int start_x, start_y, dx, dy, distance;
        decode_action(action, start_x, start_y, dx, dy, distance);
        
        stratego::Move move{start_x, start_y, start_x + dx * distance, start_y + dy * distance};

        // 1. Validate Move
        if (!board.is_legal_move(move)) {
            result.reward = -1.0f; // Penalize illegal moves
            result.observation = get_masked_observation();
            return result;
        }

        // 2. Resolve Combat or Movement
        stratego::CombatResult combat_res = board.execute_move(move);
        if (combat_res == stratego::CombatResult::FlagCaptured) {
            result.terminated = true;
            result.reward = 1.0f;
        }

        // 3. Check Truncation (Move Limit)
        if (!result.terminated && current_step >= max_steps) {
            result.truncated = true;
            // Draw game, no reward. 
        }

        result.observation = get_masked_observation();
        return result;
    }

    [[nodiscard]] size_t observation_dim() const override {
        // e.g., Width * Height * Number of Channels (Player Pieces, Enemy Pieces, Obstacles)
        int channels = 3; 
        return board_width * board_height * channels;
    }

    [[nodiscard]] size_t action_dim() const override {
        // 4 possible directions per tile
        return board_width * board_height * 4; 
    }
};