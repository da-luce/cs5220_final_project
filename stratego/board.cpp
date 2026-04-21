#include "board.h"

namespace stratego {

BoardConfig get_config_for_game_type(GameType type) {
    BoardConfig config;
    
    switch (type) {
        case GameType::Classic:
            config.width = 10;
            config.height = 10;
            config.setup_rows = 4;
            config.piece_counts = {
                {PieceType::Flag, 1},
                {PieceType::Bomb, 6},
                {PieceType::Spy, 1},
                {PieceType::Scout, 8},
                {PieceType::Miner, 5},
                {PieceType::Sergeant, 4},
                {PieceType::Lieutenant, 4},
                {PieceType::Captain, 4},
                {PieceType::Major, 3},
                {PieceType::Colonel, 2},
                {PieceType::General, 1},
                {PieceType::Marshal, 1}
            };
            config.lakes = {
                Rect{2, 4, 2, 2}, // Left lake
                Rect{6, 4, 2, 2}  // Right lake
            };
            break;

        case GameType::Quick:
            // A common smaller 8x8 variant
            config.width = 8;
            config.height = 8;
            config.setup_rows = 3;
            config.piece_counts = {
                {PieceType::Flag, 1},
                {PieceType::Bomb, 3},
                {PieceType::Spy, 1},
                {PieceType::Scout, 4},
                {PieceType::Miner, 3},
                {PieceType::Sergeant, 2},
                {PieceType::Lieutenant, 2},
                {PieceType::Captain, 2},
                {PieceType::Major, 2},
                {PieceType::Colonel, 1},
                {PieceType::General, 1},
                {PieceType::Marshal, 1}
            };
            config.lakes = {
                Rect{2, 3, 4, 2} // Single larger central lake
            };
            break;
        case GameType::Barrage:
            // 10x10 board, but significantly fewer pieces
            config.width = 10;
            config.height = 10;
            config.setup_rows = 4;
            config.piece_counts = {
                {PieceType::Flag, 1},
                {PieceType::Bomb, 2},
                {PieceType::Spy, 1},
                {PieceType::Scout, 2},
                {PieceType::Miner, 2},
                {PieceType::General, 1},
                {PieceType::Marshal, 1}
            };
            config.lakes = {
                Rect{2, 4, 2, 2},
                Rect{6, 4, 2, 2}
            };
            break;
        case GameType::Tiny:
            // A smaller 4x4 variant for quick testing
            config.width = 4;
            config.height = 4;
            config.setup_rows = 1;
            config.piece_counts = {
                {PieceType::Flag, 1},
                {PieceType::Major, 1},
                {PieceType::Captain, 1},
                {PieceType::Lieutenant, 1}
            };
            config.lakes = {}; // No lakes in the tiny variant
            break;
        default:
            throw std::invalid_argument("Unknown GameType");
    }
    return config;
}

Board::Board(const BoardConfig& cfg) : config(cfg) {
    grid.resize(config.width * config.height, Piece{PieceType::Empty, Player::None, false});
    clear(); 
}

void Board::clear() {
    // Reset all cells to empty
    for (auto& piece : grid) {
        piece.type = PieceType::Empty;
        piece.owner = Player::None;
        piece.revealed = false;
    }

    // Reset the water tiles based on the configuration
    for (const auto& lake : config.lakes) {
        for (int y = lake.y; y < lake.y + lake.h; ++y) {
            for (int x = lake.x; x < lake.x + lake.w; ++x) {
                // Bounds check just to be safe with custom configs
                if (x >= 0 && x < config.width && y >= 0 && y < config.height) {
                    grid[index(x, y)] = Piece{PieceType::Water, Player::None, true};
                }
            }
        }
    }
}

bool Board::place_piece(int x, int y, PieceType type, Player owner) {
    // 1. Check outer board boundaries
    if (x < 0 || x >= config.width || y < 0 || y >= config.height) {
        return false; 
    }

    int idx = index(x, y);

    // 2. Check if the target is a lake/obstacle (cannot place pieces on water)
    if (grid[idx].is_obstacle()) {
        return false;
    }

    // 3. Check if the cell is already occupied
    if (!grid[idx].is_empty()) {
        return false;
    }

    // Place the piece
    grid[idx] = Piece{type, owner, false};
    return true;
}

Piece Board::get_piece(int x, int y) const {
    if (x < 0 || x >= config.width || y < 0 || y >= config.height) {
        // Return a dummy empty piece if requested out of bounds
        return Piece{PieceType::Empty, Player::None, false};
    }
    return grid[index(x, y)];
}

GameHash Board::compute_hash(Player current_turn) const {
    GameHash hash = 17;
    hash = hash * 31 + static_cast<GameHash>(current_turn);
    for (const auto& piece : grid) {
        hash = hash * 31 + static_cast<GameHash>(piece.type);
        hash = hash * 31 + static_cast<GameHash>(piece.owner);
        hash = hash * 31 + static_cast<GameHash>(piece.revealed);
    }
    return hash;
}

} // namespace stratego