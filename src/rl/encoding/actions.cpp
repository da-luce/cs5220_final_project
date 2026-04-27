#include "actions.h"
#include <cmath>

namespace encoding {
namespace actions {

ActionEncoder::ActionEncoder(const stratego::BoardConfig& config) {
    // The max distance a piece can travel is the longest dimension of the board minus 1
    max_distance = std::max(config.width, config.height) - 1;
    total_channels = 4 * max_distance;
}

int ActionEncoder::get_action_channels() const {
    return total_channels;
}

int ActionEncoder::move_to_channel(const stratego::Move& move) const {
    int dx = move.end_x - move.start_x;
    int dy = move.end_y - move.start_y;

    if (dx != 0 && dy != 0) throw std::invalid_argument("Diagonal moves not allowed");
    if (dx == 0 && dy == 0) throw std::invalid_argument("Stationary moves not allowed");

    int distance = std::max(std::abs(dx), std::abs(dy));
    if (distance > max_distance) throw std::invalid_argument("Move distance exceeds board dimensions");

    // Convert 1-indexed distance to 0-indexed offset for the channel bucket
    int dist_offset = distance - 1;

    // Buckets:
    // 0: North
    // 1: South
    // 2: East
    // 3: West
    if (dy < 0) {
        return (0 * max_distance) + dist_offset; // North
    } 
    else if (dy > 0) {
        return (1 * max_distance) + dist_offset; // South
    } 
    else if (dx > 0) {
        return (2 * max_distance) + dist_offset; // East
    } 
    else {
        return (3 * max_distance) + dist_offset; // West
    }
}

stratego::Move ActionEncoder::channel_to_move(int channel, int source_x, int source_y) const {
    if (channel < 0 || channel >= total_channels) {
        throw std::out_of_range("Invalid action channel index");
    }

    int target_x = source_x;
    int target_y = source_y;

    // Determine direction bucket and distance
    int direction = channel / max_distance; 
    int distance = (channel % max_distance) + 1; 

    switch (direction) {
        case 0: target_y -= distance; break; // North
        case 1: target_y += distance; break; // South
        case 2: target_x += distance; break; // East
        case 3: target_x -= distance; break; // West
    }

    return stratego::Move{source_x, source_y, target_x, target_y};
}

} // namespace actions
} // namespace encoding