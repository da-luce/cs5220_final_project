#pragma once
#include "board.h"
#include <vector>
#include <string>
#include <cstddef>

namespace stratego{

struct Move {
    int start_x, start_y;
    int end_x, end_y;

    // Record the exact pieces involved so bots can be informed after combat
    PieceType attacker_type = PieceType::Empty;
    PieceType defender_type = PieceType::Empty;
};

struct GameState {
    Board board;
    int max_moves{2000}; // Default max moves before declaring a draw
    Player current_turn{Player::Red};
    int move_count{0};
    std::vector<Move> move_history;
    // Stores board states after moves that revealed nothing, for detecting loops
    // Used to implement the More-Squares Rule
    // https://web.archive.org/web/20110123114925/http://www.strategousa.org/wiki/index.php/2010_Computer_Stratego_World_Championship
    std::vector<GameHash> chase_hashes;
};

namespace state {

enum class SetupType {
    Default,
    Random,
    Probabilistic
};

using GameBinary = std::vector<std::byte>;

// Create a new game state with the given board config and setup type
GameState initialize(BoardConfig board_config, SetupType setup, int max_moves);

// Serialization
GameBinary serialize(GameState state);
GameState deserialize(const GameBinary& data);

// Returns a masked view of the game state for the given player
inline GameState get_masked_view(const GameState& game_state, Player player) {
    GameState view = game_state;
    view.board = game_state.board.get_masked_view(player);
    return view;
}

} // namespace state

} // namespace stratego