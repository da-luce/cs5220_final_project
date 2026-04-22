#include "game.h"
#include <random>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace stratego {

Game::Game(const BoardConfig& config, int max_moves)
    : board(config), max_moves(max_moves) {
    // board is already initialized via the initializer list
}

void Game::initialize_game(SetupType setup) {
    board.clear();
    move_count = 0;
    current_turn = Player::Red;
    move_history.clear();
    chase_hashes.clear();

    if (setup == SetupType::Random || setup == SetupType::Default) {
        // 1. Flatten the piece counts into a list of pieces
        std::vector<PieceType> army;
        for (const auto& [type, count] : board.config.piece_counts) {
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
        for (int y = 0; y < board.config.setup_rows; ++y) {
            for (int x = 0; x < board.config.width; ++x) {
                if (red_idx < red_army.size()) {
                    place_piece(x, y, red_army[red_idx++], Player::Red);
                }
            }
        }

        // 4. Place Blue Army (Bottom of board: y from height - setup_rows to height - 1)
        int blue_idx = 0;
        for (int y = board.config.height - board.config.setup_rows; y < board.config.height; ++y) {
            for (int x = 0; x < board.config.width; ++x) {
                if (blue_idx < blue_army.size()) {
                    place_piece(x, y, blue_army[blue_idx++], Player::Blue);
                }
            }
        }
    }
}

bool Game::place_piece(int x, int y, PieceType type, Player owner) {
    // Delegate entirely to the Board's boundary and obstacle checking
    return board.place_piece(x, y, type, owner);
}

CombatResult Game::execute_move(const Move& move) {
    if (!is_legal_move(move)) {
        return CombatResult::InvalidMove;
    }

    // Delegate the physical mechanics to the stateless Engine
    CombatResult result = Engine::execute_move(this->board, move);

    // Update match history
    move_history.push_back(move);
    move_count++;
    
    // Swap the turn to the opponent
    current_turn = (current_turn == Player::Red) ? Player::Blue : Player::Red;
    
    // Log the resulting hash for repetition checking (More-Squares Rule)
    chase_hashes.push_back(get_current_hash());

    // Match-Level Limits
    if (max_moves > 0 && move_count >= max_moves) {
        return CombatResult::Draw;
    }

    return result;
}

// --- Binary Serialization ---

GameBinary Game::serialize() const {
    GameBinary data;

    auto append_data = [&data](const void* ptr, size_t size) {
        const std::byte* byte_ptr = static_cast<const std::byte*>(ptr);
        data.insert(data.end(), byte_ptr, byte_ptr + size);
    };

    // 1. Write basic state
    append_data(&current_turn, sizeof(current_turn));
    append_data(&move_count, sizeof(move_count));

    // 2. Write board state (Assuming grid size doesn't change from BoardConfig)
    size_t grid_size = board.grid.size();
    append_data(&grid_size, sizeof(grid_size));
    if (grid_size > 0) {
        append_data(board.grid.data(), grid_size * sizeof(Piece));
    }

    // 3. Write Move History
    size_t history_size = move_history.size();
    append_data(&history_size, sizeof(history_size));
    if (history_size > 0) {
        append_data(move_history.data(), history_size * sizeof(Move));
    }

    // 4. Write Chase Hashes
    size_t hash_size = chase_hashes.size();
    append_data(&hash_size, sizeof(hash_size));
    if (hash_size > 0) {
        append_data(chase_hashes.data(), hash_size * sizeof(GameHash));
    }

    return data;
}

void Game::deserialize(const GameBinary& data) {
    size_t offset = 0;

    auto read_data = [&data, &offset](void* dest, size_t size) {
        if (offset + size > data.size()) {
            throw std::runtime_error("Deserialization failed: out of bounds");
        }
        std::memcpy(dest, data.data() + offset, size);
        offset += size;
    };

    // 1. Read basic state
    read_data(&current_turn, sizeof(current_turn));
    read_data(&move_count, sizeof(move_count));

    // 2. Read board state
    size_t grid_size = 0;
    read_data(&grid_size, sizeof(grid_size));
    if (grid_size > 0) {
        board.grid.resize(grid_size);
        read_data(board.grid.data(), grid_size * sizeof(Piece));
    }

    // 3. Read Move History
    size_t history_size = 0;
    read_data(&history_size, sizeof(history_size));
    if (history_size > 0) {
        move_history.resize(history_size);
        read_data(move_history.data(), history_size * sizeof(Move));
    } else {
        move_history.clear();
    }

    // 4. Read Chase Hashes
    size_t hash_size = 0;
    read_data(&hash_size, sizeof(hash_size));
    if (hash_size > 0) {
        chase_hashes.resize(hash_size);
        read_data(chase_hashes.data(), hash_size * sizeof(GameHash));
    } else {
        chase_hashes.clear();
    }
}

} // namespace stratego