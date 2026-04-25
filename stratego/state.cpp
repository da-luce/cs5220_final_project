#include "state.h"
#include "setup.h"
#include <random>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace stratego {

GameState state::initialize(BoardConfig board_config, SetupType setup, int max_moves) {
    GameState state;
    state.board = Board(board_config);
    state.max_moves = max_moves;
    state.move_count = 0;
    state.current_turn = Player::Red;
    state.move_history.clear();
    state.chase_hashes.clear();

    if (setup == SetupType::Random || setup == SetupType::Default) {
        setup::generate_random_setup(state.board, Player::Red);
        setup::generate_random_setup(state.board, Player::Blue);
    }

    if (setup == SetupType::Probabilistic) {
        // Use setup algorithm based on expert distributions
        setup::generate_probabilistic_setup(state.board, Player::Red);
        setup::generate_probabilistic_setup(state.board, Player::Blue);
    }

    return state;
}

state::GameBinary state::serialize(GameState state) {
    state::GameBinary data;

    auto append_data = [&data](const void* ptr, size_t size) {
        const std::byte* byte_ptr = static_cast<const std::byte*>(ptr);
        data.insert(data.end(), byte_ptr, byte_ptr + size);
    };

    // 1. Write basic state
    append_data(&state.current_turn, sizeof(state.current_turn));
    append_data(&state.move_count, sizeof(state.move_count));

    // 2. Write board state (Assuming grid size doesn't change from BoardConfig)
    size_t grid_size = state.board.grid.size();
    append_data(&grid_size, sizeof(grid_size));
    if (grid_size > 0) {
        append_data(state.board.grid.data(), grid_size * sizeof(Piece));
    }

    // 3. Write Move History
    size_t history_size = state.move_history.size();
    append_data(&history_size, sizeof(history_size));
    if (history_size > 0) {
        append_data(state.move_history.data(), history_size * sizeof(Move));
    }

    // 4. Write Chase Hashes
    size_t hash_size = state.chase_hashes.size();
    append_data(&hash_size, sizeof(hash_size));
    if (hash_size > 0) {
        append_data(state.chase_hashes.data(), hash_size * sizeof(GameHash));
    }

    return data;
}

GameState state::deserialize(const state::GameBinary& data) {
    GameState state;
    size_t offset = 0;

    auto read_data = [&data, &offset](void* dest, size_t size) {
        if (offset + size > data.size()) {
            throw std::runtime_error("Deserialization failed: out of bounds");
        }
        std::memcpy(dest, data.data() + offset, size);
        offset += size;
    };

    // 1. Read basic state
    read_data(&state.current_turn, sizeof(state.current_turn));
    read_data(&state.move_count, sizeof(state.move_count));

    // 2. Read board state
    size_t grid_size = 0;
    read_data(&grid_size, sizeof(grid_size));
    if (grid_size > 0) {
        state.board.grid.resize(grid_size);
        read_data(state.board.grid.data(), grid_size * sizeof(Piece));
    }

    // 3. Read Move History
    size_t history_size = 0;
    read_data(&history_size, sizeof(history_size));
    if (history_size > 0) {
        state.move_history.resize(history_size);
        read_data(state.move_history.data(), history_size * sizeof(Move));
    } else {
        state.move_history.clear();
    }

    // 4. Read Chase Hashes
    size_t hash_size = 0;
    read_data(&hash_size, sizeof(hash_size));
    if (hash_size > 0) {
        state.chase_hashes.resize(hash_size);
        read_data(state.chase_hashes.data(), hash_size * sizeof(GameHash));
    } else {
        state.chase_hashes.clear();
    }

    return state;
}

} // namespace stratego