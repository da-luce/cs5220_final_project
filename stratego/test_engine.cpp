#include <gtest/gtest.h>
#include "engine.h"
#include <string>

using namespace stratego;

#include "state.h"
// We could hand write our own hashes and move history, but it's easier to just
// execute some moves and then save the resulting state.

TEST(EngineTest, OutOfBounds) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Sergeant, Player::Red);

    // Illegal moves outside the board dimensions
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, -1, 0}));
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, -1}));
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 10, 0}));
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 10}));
}

TEST(EngineTest, ImmobilePieces) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Flag, Player::Red);
    state.board.place_piece(1, 0, PieceType::Bomb, Player::Red);

    // Flags and Bombs cannot be moved
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 1}));
    EXPECT_FALSE(Engine::is_legal_move(state, {1, 0, 1, 1}));
}

TEST(EngineTest, StandardMoves) {
    GameState state;
    state.board.place_piece(1, 1, PieceType::Captain, Player::Red);

    // Standard pieces can move exactly 1 tile orthogonally
    EXPECT_TRUE(Engine::is_legal_move(state, {1, 1, 1, 2}));
    EXPECT_TRUE(Engine::is_legal_move(state, {1, 1, 2, 1}));

    // Standard pieces cannot move > 1 tile
    EXPECT_FALSE(Engine::is_legal_move(state, {1, 1, 1, 3}));
    
    // Standard pieces cannot move diagonally
    EXPECT_FALSE(Engine::is_legal_move(state, {1, 1, 2, 2}));

    // Standard pieces cannot move to the same square
    EXPECT_FALSE(Engine::is_legal_move(state, {1, 1, 1, 1}));

    // Moving an empty square is illegal
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 1}));
}

TEST(EngineTest, ScoutMoves) {
    GameState state;
    
    // In a standard 10x10, water is at x in [2,3] and [6,7], y in [4,5]
    state.board.place_piece(2, 2, PieceType::Scout, Player::Red);

    // 1. Moving across water is illegal
    EXPECT_FALSE(Engine::is_legal_move(state, {2, 2, 2, 6})); 
    
    // Moving INTO water is also illegal
    EXPECT_FALSE(Engine::is_legal_move(state, {2, 2, 2, 4}));

    // 2. Crossing multiple enemy pieces is illegal
    state.board.place_piece(0, 0, PieceType::Scout, Player::Red);
    state.board.place_piece(0, 1, PieceType::Sergeant, Player::Blue);
    state.board.place_piece(0, 2, PieceType::Sergeant, Player::Blue);
    
    // Attempt to move past them to an empty square
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 3}));
    // Attempt to attack the second one by jumping the first
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 2}));

    // 3. Crossing your own pieces is illegal
    state.board.place_piece(9, 0, PieceType::Scout, Player::Red);
    state.board.place_piece(9, 1, PieceType::Spy, Player::Red);
    EXPECT_FALSE(Engine::is_legal_move(state, {9, 0, 9, 2}));

    // 4. Valid long-distance attack
    state.board.place_piece(5, 0, PieceType::Scout, Player::Red);
    state.board.place_piece(5, 3, PieceType::Major, Player::Blue);
    EXPECT_TRUE(Engine::is_legal_move(state, {5, 0, 5, 3})); // Legal attack
}

TEST(EngineTest, CombatResolutions) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Miner, Player::Red);
    state.board.place_piece(0, 1, PieceType::Bomb, Player::Blue);
    
    EXPECT_TRUE(Engine::is_legal_move(state, {0, 0, 0, 1}));
    CombatResult res1 = Engine::execute_move(state, {0, 0, 0, 1});
    EXPECT_EQ(res1, CombatResult::AttackerWins); // Miner defuses bomb

    // Next turn is Blue's
    state.board.place_piece(9, 9, PieceType::Spy, Player::Blue);
    state.board.place_piece(9, 8, PieceType::Marshal, Player::Red);
    
    CombatResult res2 = Engine::execute_move(state, {9, 9, 9, 8});
    EXPECT_EQ(res2, CombatResult::AttackerWins); // Spy kills Marshal
}

TEST(EngineTest, SerializeDeserialize) {
    GameState state1;
    state1.board.place_piece(0, 7, PieceType::Marshal, Player::Red);
    state1.board.place_piece(0, 6, PieceType::Spy, Player::Red); // Attacker
    state1.board.place_piece(0, 5, PieceType::General, Player::Blue); // Defender
    
    // Make a move to change turn and reveal a piece.
    // Red Spy (1) attacks Blue General (9). General wins, Spy is removed.
    Engine::execute_move(state1, {0, 6, 0, 5}); 
    
    // Save the state after the move
    state::GameBinary data = state::serialize(state1);
    EXPECT_FALSE(data.empty());

    // Load the state into a new, empty board
    GameState state2;
    EXPECT_NO_THROW(state2 = state::deserialize(data));

    // Compare the two boards to ensure they are identical
    EXPECT_EQ(state1.current_turn, state2.current_turn);
    EXPECT_EQ(state2.current_turn, Player::Blue); // Red moved, so it's Blue's turn
    
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            Piece p1 = state1.board.get_piece(x, y);
            Piece p2 = state2.board.get_piece(x, y);
            EXPECT_EQ(p1.type, p2.type);
            EXPECT_EQ(p1.owner, p2.owner);
            EXPECT_EQ(p1.revealed, p2.revealed);
        }
    }
}

TEST(EngineTest, TwoSquaresRule) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Scout, Player::Red);
    state.board.place_piece(9, 9, PieceType::Scout, Player::Blue);

    EXPECT_EQ(Engine::execute_move(state, {0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state, {0, 1, 0, 0}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {9, 8, 9, 9}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state, {0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    
    // R attempts to return to (0,0) again, triggering the back-and-forth loop limit
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 1, 0, 0})); 
}

TEST(EngineTest, MoreSquaresRule) {
    GameState state;
    state.board.place_piece(1, 1, PieceType::Scout, Player::Red);
    state.board.place_piece(2, 2, PieceType::Scout, Player::Blue);

    EXPECT_EQ(Engine::execute_move(state, {1, 1, 1, 2}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {2, 2, 2, 3}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state, {1, 2, 2, 2}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {2, 3, 1, 3}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state, {2, 2, 2, 3}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {1, 3, 1, 2}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state, {2, 3, 1, 3}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state, {1, 2, 2, 2}), CombatResult::MovedToEmpty); // B
    
    // R chases 1,3 -> 1,2, recreating a previously visited board state hash!
    EXPECT_FALSE(Engine::is_legal_move(state, {1, 3, 1, 2}));
}

TEST(EngineTest, TurnAndHistoryManagement) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Scout, Player::Red);
    state.board.place_piece(0, 9, PieceType::Scout, Player::Blue);

    // Initial state
    EXPECT_EQ(state.current_turn, Player::Red);
    EXPECT_EQ(state.move_count, 0);

    // Make a move
    CombatResult res1 = Engine::execute_move(state, {0, 0, 0, 1});
    EXPECT_EQ(res1, CombatResult::MovedToEmpty);

    // State should update correctly
    EXPECT_EQ(state.current_turn, Player::Blue);
    EXPECT_EQ(state.move_count, 1);
    EXPECT_EQ(state.move_history.size(), 1);
    EXPECT_EQ(state.chase_hashes.size(), 1);
    
    // Move history should exactly match what was played
    Move played = state.move_history[0];
    EXPECT_EQ(played.start_x, 0);
    EXPECT_EQ(played.start_y, 0);
    EXPECT_EQ(played.end_x, 0);
    EXPECT_EQ(played.end_y, 1);
}


TEST(EngineTest, FlippedMoveLogic) {
    BoardConfig config = get_config_for_game_type(GameType::Classic); // 10x10
    
    // A move from the top-left corner to one tile down
    Move original{0, 0, 0, 1};
    
    // Flip it for the other player's perspective
    Move flipped = Engine::get_flipped_move(config, original);
    
    // In a 10x10, (0,0) flips to (9,9)
    // The destination (0,1) flips to (9,8)
    EXPECT_EQ(flipped.start_x, 9);
    EXPECT_EQ(flipped.start_y, 9);
    EXPECT_EQ(flipped.end_x, 9);
    EXPECT_EQ(flipped.end_y, 8);
    
    // Symmetry check: Flipping a flipped move should yield the original
    Move back_to_original = Engine::get_flipped_move(config, flipped);
    EXPECT_EQ(back_to_original.start_x, original.start_x);
    EXPECT_EQ(back_to_original.end_y, original.end_y);
}

TEST(EngineTest, FlippedMoveQuickConfig) {
    BoardConfig config = get_config_for_game_type(GameType::Quick); // 8x8
    
    Move original{2, 3, 3, 3}; // A move in the middle-ish
    Move flipped = Engine::get_flipped_move(config, original);
    
    // (8-1-2) = 5, (8-1-3) = 4
    EXPECT_EQ(flipped.start_x, 5);
    EXPECT_EQ(flipped.start_y, 4);
}

TEST(EngineTest, DirectGameStateUsage) {
    GameState state;
    state.board = Board(get_config_for_game_type(GameType::Tiny));
    state.current_turn = Player::Red;
    state.board.place_piece(0, 0, PieceType::Scout, Player::Red);
    
    EXPECT_TRUE(Engine::is_legal_move(state, {0, 0, 0, 1}));
    EXPECT_TRUE(Engine::is_legal_move(state, {0, 0, 0, 3}));
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, 4}));
    EXPECT_FALSE(Engine::is_legal_move(state, {0, 0, 0, -1}));
}