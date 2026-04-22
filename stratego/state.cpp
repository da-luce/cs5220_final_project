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
        // 1. Flatten the piece counts into a list of pieces
        std::vector<PieceType> army;
        for (const auto& [type, count] : state.board.config.piece_counts) {
            for (int i = 0; i < count; ++i) {
                army.push_back(type);
            }
        }

        // 2. Shuffle for random placement
        std::random_device rd;
        std::mt19937 g(rd());
        
        std::vector<PieceType> red_army = army;
        std::vector<PieceType> blue_army = army;
        std::shuffle(red_army.begin(), red_army.end(), g);
        std::shuffle(blue_army.begin(), blue_army.end(), g);

        // 3. Place Red Army (Top of board: y from 0 to setup_rows - 1)
        int red_idx = 0;
        for (int y = 0; y < state.board.config.setup_rows; ++y) {
            for (int x = 0; x < state.board.config.width; ++x) {
                if (red_idx < red_army.size()) {
                    state.board.place_piece(x, y, red_army[red_idx++], Player::Red);
                }
            }
        }

        // 4. Place Blue Army (Bottom of board: y from height - setup_rows to height - 1)
        int blue_idx = 0;
        for (int y = state.board.config.height - state.board.config.setup_rows; y < state.board.config.height; ++y) {
            for (int x = 0; x < state.board.config.width; ++x) {
                if (blue_idx < blue_army.size()) {
                    state.board.place_piece(x, y, blue_army[blue_idx++], Player::Blue);
                }
            }
        }
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