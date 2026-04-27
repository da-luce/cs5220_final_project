#pragma once
#include "stratego/state.h"
#include "stratego/board.h"
#include <stdexcept>
#include <algorithm>

/** Action encoding scheme:
 *
 * The action space is represented as a spatial tensor of shape (D, H, W),
 * where H and W are the board dimensions, and D represents the move type.
 * The spatial dimensions (h, w) correspond to the SOURCE square of the piece being moved.
 *
 * Let D_max = max(H, W) - 1. This is the absolute maximum distance a piece (e.g., a Scout)
 * can travel in a single straight line from one edge of the board to the other.
 * We will have a total of (4 * D_max) channels in our action probability tensor:
 *
 * [0            to     D_max - 1] : Move North (distance 1 to D_max)
 * [D_max        to 2 * D_max - 1] : Move South (distance 1 to D_max)
 * [2 * D_max    to 3 * D_max - 1] : Move East  (distance 1 to D_max)
 * [3 * D_max    to 4 * D_max - 1] : Move West  (distance 1 to D_max)
 *
 * For example, on a standard 10x10 board, D_max = 9. 
 * Channel 0 means "Move North 1", Channel 8 means "Move North 9".
 * Channel 18 means "Move East 1".
 *
 * To map the network's output to a physical move, we look at the channel `d` 
 * and the spatial coordinates `(x, y)`. This tells us exactly what piece to pick up, 
 * what direction to move it, and how far.
 *
 * NOTE: Because the vast majority of these (4 * D_max * H * W) actions are illegal 
 * for any given state, a Legal Move Mask (setting illegal moves to -1e9) must be 
 * applied to the raw network logits before applying Softmax during RL training or MCTS.
 */

namespace encoding {
namespace actions {

class ActionEncoder {
private:
    int max_distance;
    int total_channels;

public:
    // Initialize the encoder using the board configuration
    explicit ActionEncoder(const stratego::BoardConfig& config);

    // Returns the total number of channels (needed to initialize the Policy Head)
    int get_action_channels() const;

    // Maps a move to a channel index [0, total_channels - 1]
    int move_to_channel(const stratego::Move& move) const;

    // Maps a channel index and source coordinate back to a Move
    stratego::Move channel_to_move(int channel, int source_x, int source_y) const;
};

} // namespace actions
} // namespace encoding