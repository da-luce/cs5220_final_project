#pragma once
#include "board.h"
#include "engine.h" 
#include <vector>
#include <string>

namespace stratego {

enum class SetupType {
    Default,
    Random,
    Probabilistic
};

class Game {
private:
    Board board;
    Player current_turn{Player::Red};
    int move_count{0};
    int max_moves{2000}; // Default max moves before declaring a draw
    
    std::vector<Move> move_history;

    // Stores board states after moves that revealed nothing, for detecting loops
    // Used to implement the More-Squares Rule
    // https://web.archive.org/web/20110123114925/http://www.strategousa.org/wiki/index.php/2010_Computer_Stratego_World_Championship
    std::vector<GameHash> chase_hashes;

public:
    explicit Game(const BoardConfig& config = get_config_for_game_type(GameType::Classic), int max_moves = 2000);

    // --- Setup Phase ---
    void initialize_game(SetupType setup = SetupType::Default);
    bool place_piece(int x, int y, PieceType type, Player owner);

    // --- State Hashing ---
    [[nodiscard]] GameHash get_current_hash() const {
        return board.compute_hash(current_turn);
    }

    // --- Core Game Loop (Delegates strictly to Engine) ---
    [[nodiscard]] inline bool is_legal_move(const Move& move) const {
        return Engine::is_legal_move(board, move, current_turn, chase_hashes, move_history);
    }

    [[nodiscard]] inline std::vector<Move> get_all_legal_moves(Player player) const {
        return Engine::get_all_legal_moves(board, player, chase_hashes, move_history);
    }

    [[nodiscard]] inline std::vector<Move> get_legal_moves_for_piece(int x, int y) const {
        return Engine::get_legal_moves_for_piece(board, x, y, chase_hashes, move_history);
    }

    // Executes the move via Engine, updates histories, and swaps the turn
    CombatResult execute_move(const Move& move);

    // --- File I/O ---
    [[nodiscard]] bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);

    // --- Frontend Getters ---
    [[nodiscard]] Player get_current_turn() const { return current_turn; }
    [[nodiscard]] const Board& get_board() const { return board; }
    [[nodiscard]] int get_move_count() const { return move_count; }
    [[nodiscard]] const std::vector<Move>& get_history() const { return move_history; }
    [[nodiscard]] const std::vector<GameHash>& get_chase_hashes() const { return chase_hashes; }

    // Expose engine flip utility for opponent view generation
    [[nodiscard]] static Move get_flipped_move(const BoardConfig& config, const Move& move) {
        return Engine::get_flipped_move(config, move);
    }
};

} // namespace stratego