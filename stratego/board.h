#pragma once
#include <vector>
#include <map>
#include <cstdint>

namespace stratego {

using GameHash = std::uint64_t;

enum class Player {
    Red,
    Blue,
    None
};

// Using numbering where 10 is highest & strongest piece
enum class PieceType {
    Empty = 0,
    Spy = 1,
    Scout = 2,
    Miner = 3,
    Sergeant = 4,
    Lieutenant = 5,
    Captain = 6,
    Major = 7,
    Colonel = 8,
    General = 9,
    Marshal = 10,
    Bomb = 11,
    Flag = 12,
    Water = 13,
    Hidden = 14     // Not for use on the board, but for encoding hidden enemy pieces in observations
};

using PieceCounts = std::map<PieceType, int>;

struct Rect {
    int x;
    int y;
    int w;
    int h;

    bool contains(int px, int py) const {
        return px >= x && px < x + w &&
               py >= y && py < y + h;
    }
};

// Defines a game type
struct BoardConfig {
    int width;
    int height;
    int setup_rows;             // Number of rows each player uses for setup (starting from their edge)
    PieceCounts piece_counts;   // The piece counts per player (total pieces = 2 * sum of counts in this map) 
    std::vector<Rect> lakes;    // The positions of lakes on the board 
};

// A few common modes are pre-defined
enum class GameType {
    Classic,
    Tiny,
    Quick,
    Barrage
};

BoardConfig get_config_for_game_type(GameType type);

struct Piece {
    PieceType type{PieceType::Empty};
    Player owner{Player::None};
    bool revealed{false}; // True if piece survived a combat

    [[nodiscard]] bool is_empty() const { 
        return type == PieceType::Empty; 
    }
    
    [[nodiscard]] bool is_obstacle() const { 
        return type == PieceType::Water; 
    }
    
    [[nodiscard]] bool is_mobile() const { 
        return type != PieceType::Empty && 
               type != PieceType::Water && 
               type != PieceType::Bomb && 
               type != PieceType::Flag; 
    }
};

class Board {
public:
    BoardConfig config;
    std::vector<Piece> grid;

    Board(const BoardConfig& config = get_config_for_game_type(GameType::Classic));

    // Remove all pieces
    void clear();

    // Set a piece
    // Top left is (0,0), x increases left->right, y increases top->bottom
    bool place_piece(int x, int y, PieceType type, Player owner);

    // Returns a sanitized copy of the board from the perspective of the given player
    [[nodiscard]] Board get_player_view(Player player) const;

    // Returns a flipped version of the board (for the opponent's perspective)
    [[nodiscard]] Board get_flipped_board() const;

    // Map 2D coordinates to 1D vector
    [[nodiscard]] inline int index(int x, int y) const { return y * config.width + x; }
    [[nodiscard]] int get_width() const { return config.width; }
    [[nodiscard]] int get_height() const { return config.height; }
    [[nodiscard]] const BoardConfig& get_config() const { return config; }
    [[nodiscard]] Piece get_piece(int x, int y) const;
    [[nodiscard]] GameHash compute_hash(Player current_turn) const;
};

} // namespace stratego