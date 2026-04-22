#include "features.h"
#include <stdexcept>
#include <algorithm>

BoardFeatures get_board_encoding(const stratego::GameState& state, stratego::Player perspective_player) {
    const int w = state.board.get_width();
    const int h = state.board.get_height();
    const int spatial_size = w * h;

    // 1. Map piece types to channel offsets (0 to N-1)
    int piece_to_base_channel[15];
    std::fill(std::begin(piece_to_base_channel), std::end(piece_to_base_channel), -1);

    int num_playable = 0;
    for (const auto& [ptype, count] : state.board.config.piece_counts) {
        piece_to_base_channel[static_cast<int>(ptype)] = num_playable++;
    }

    const int total_channels = (2 * num_playable) + 2;
    const int unrevealed_offset = num_playable * spatial_size;
    const int revealed_start_channel = num_playable + 1;
    const int obstacle_offset = (total_channels - 1) * spatial_size;

    // 2. Initialize flat vector with zeros
    BoardFeatures obs(total_channels * spatial_size, 0.0f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            stratego::Piece p = state.board.get_piece(x, y);
            if (p.is_empty()) continue;

            // Perspective Flip
            int view_x = (perspective_player == stratego::Player::Red) ? x : (w - 1 - x);
            int view_y = (perspective_player == stratego::Player::Red) ? y : (h - 1 - y);
            int spatial_idx = (view_y * w) + view_x;

            if (p.is_obstacle()) {
                obs[obstacle_offset + spatial_idx] = 1.0f;
                continue;
            }

            int piece_val = static_cast<int>(p.type);
            if (piece_val == 14) { // Hidden
                obs[unrevealed_offset + spatial_idx] = 1.0f;
                continue;
            }

            int base_idx = piece_to_base_channel[piece_val];
            if (base_idx == -1) throw std::runtime_error("Piece not in config!");

            int final_channel;
            if (p.owner == perspective_player) {
                final_channel = base_idx;
            } else if (p.revealed) {
                final_channel = revealed_start_channel + base_idx;
            } else {
                obs[unrevealed_offset + spatial_idx] = 1.0f;
                continue;
            }

            obs[(final_channel * spatial_size) + spatial_idx] = 1.0f;
        }
    }
    return obs;
}