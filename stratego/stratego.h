// Generic Stratego environment implementation in C++. We implement main game
// logic here and use functions in other modules as needed.

#pragma once
#include <vector>
#include <string>

namespace stratego {

enum class Player {
    Red,
    Blue,
    None
};

enum class GameType {
    Normal,
    Tiny,
    Quick
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
    Water = 13
};

struct Piece {
    PieceType type{PieceType::Empty};
    Player owner{Player::None};
    bool revealed{false}; // Crucial for RL: True if piece survived a combat

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

struct Move {
    int start_x, start_y;
    int end_x, end_y;
};

// All possible results of a move, makes implementing the front end simpler
enum class CombatResult {
    MovedToEmpty,
    AttackerWins,
    DefenderWins,
    BothDestroyed,
    FlagCaptured,
    InvalidMove
};

class Board {
private:
    int width;
    int height;
    std::vector<Piece> grid;
    Player current_turn;

    // Map 2D coordinates to 1D vector
    [[nodiscard]] inline int index(int x, int y) const { return y * width + x; }

public:
    Board(int w = 10, int h = 10);

    // Resets board with standard water placement
    void initialize_empty(); 

    // Initializes board layout and randomized piece placement for a specific game type
    void initialize_game(GameType type);

    // Set pieces during setup phase
    bool place_piece(int x, int y, PieceType type, Player owner);

    // Validates a move
    [[nodiscard]] bool is_legal_move(const Move& move) const;

    // Executes the move, handles combat, updates "revealed" statuses, and swaps turn
    CombatResult execute_move(const Move& move);

    // Frontend getters
    [[nodiscard]] int get_width() const { return width; }
    [[nodiscard]] int get_height() const { return height; }
    [[nodiscard]] Piece get_piece(int x, int y) const;
    [[nodiscard]] Player get_current_turn() const { return current_turn; }
    [[nodiscard]] std::vector<Move> get_all_legal_moves(Player player) const;
    [[nodiscard]] std::vector<Move> get_legal_moves_for_piece(Player player, int x, int y) const;

    // Save/Load game state to a binary file
    [[nodiscard]] bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);
};

} // namespace stratego