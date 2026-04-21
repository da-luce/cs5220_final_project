#include <gtest/gtest.h>
#include "engine.h"
#include <cstdio>   // For std::remove
#include <string>

using namespace stratego;

#include "game.h"
// We could hand write our own hashes and move history, but it's easier to just
// execute some moves and then save the resulting state.

TEST(EngineTest, OutOfBounds) {
    Game game;
    game.place_piece(0, 0, PieceType::Sergeant, Player::Red);

    // Illegal moves outside the board dimensions
    EXPECT_FALSE(game.is_legal_move({0, 0, -1, 0}));
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, -1}));
    EXPECT_FALSE(game.is_legal_move({0, 0, 10, 0}));
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, 10}));
}

TEST(EngineTest, ImmobilePieces) {
    Game game;
    game.place_piece(0, 0, PieceType::Flag, Player::Red);
    game.place_piece(1, 0, PieceType::Bomb, Player::Red);

    // Flags and Bombs cannot be moved
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, 1}));
    EXPECT_FALSE(game.is_legal_move({1, 0, 1, 1}));
}

TEST(EngineTest, StandardMoves) {
    Game game;
    game.place_piece(1, 1, PieceType::Captain, Player::Red);

    // Standard pieces can move exactly 1 tile orthogonally
    EXPECT_TRUE(game.is_legal_move({1, 1, 1, 2}));
    EXPECT_TRUE(game.is_legal_move({1, 1, 2, 1}));

    // Standard pieces cannot move > 1 tile
    EXPECT_FALSE(game.is_legal_move({1, 1, 1, 3}));
    
    // Standard pieces cannot move diagonally
    EXPECT_FALSE(game.is_legal_move({1, 1, 2, 2}));

    // Standard pieces cannot move to the same square
    EXPECT_FALSE(game.is_legal_move({1, 1, 1, 1}));

    // Moving an empty square is illegal
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, 1}));
}

TEST(EngineTest, ScoutMoves) {
    Game game;
    
    // In a standard 10x10, water is at x in [2,3] and [6,7], y in [4,5]
    game.place_piece(2, 2, PieceType::Scout, Player::Red);

    // 1. Moving across water is illegal
    EXPECT_FALSE(game.is_legal_move({2, 2, 2, 6})); 
    
    // Moving INTO water is also illegal
    EXPECT_FALSE(game.is_legal_move({2, 2, 2, 4}));

    // 2. Crossing multiple enemy pieces is illegal
    game.place_piece(0, 0, PieceType::Scout, Player::Red);
    game.place_piece(0, 1, PieceType::Sergeant, Player::Blue);
    game.place_piece(0, 2, PieceType::Sergeant, Player::Blue);
    
    // Attempt to move past them to an empty square
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, 3}));
    // Attempt to attack the second one by jumping the first
    EXPECT_FALSE(game.is_legal_move({0, 0, 0, 2}));

    // 3. Crossing your own pieces is illegal
    game.place_piece(9, 0, PieceType::Scout, Player::Red);
    game.place_piece(9, 1, PieceType::Spy, Player::Red);
    EXPECT_FALSE(game.is_legal_move({9, 0, 9, 2}));

    // 4. Valid long-distance attack
    game.place_piece(5, 0, PieceType::Scout, Player::Red);
    game.place_piece(5, 3, PieceType::Major, Player::Blue);
    EXPECT_TRUE(game.is_legal_move({5, 0, 5, 3})); // Legal attack
}

TEST(EngineTest, CombatResolutions) {
    Game game;
    game.place_piece(0, 0, PieceType::Miner, Player::Red);
    game.place_piece(0, 1, PieceType::Bomb, Player::Blue);
    
    EXPECT_TRUE(game.is_legal_move({0, 0, 0, 1}));
    CombatResult res1 = game.execute_move({0, 0, 0, 1});
    EXPECT_EQ(res1, CombatResult::AttackerWins); // Miner defuses bomb

    // Next turn is Blue's
    game.place_piece(9, 9, PieceType::Spy, Player::Blue);
    game.place_piece(9, 8, PieceType::Marshal, Player::Red);
    
    CombatResult res2 = game.execute_move({9, 9, 9, 8});
    EXPECT_EQ(res2, CombatResult::AttackerWins); // Spy kills Marshal
}

TEST(EngineTest, SaveLoad) {
    const std::string filename = "test_save.bin";
    
    Game game1;
    game1.place_piece(0, 7, PieceType::Marshal, Player::Red);
    game1.place_piece(0, 6, PieceType::Spy, Player::Red); // Attacker
    game1.place_piece(0, 5, PieceType::General, Player::Blue); // Defender
    
    // Make a move to change turn and reveal a piece.
    // Red Spy (1) attacks Blue General (9). General wins, Spy is removed.
    game1.execute_move({0, 6, 0, 5}); 
    
    // Save the state after the move
    EXPECT_TRUE(game1.save_to_file(filename));

    // Load the state into a new, empty board
    Game game2;
    EXPECT_TRUE(game2.load_from_file(filename));

    // Compare the two boards to ensure they are identical
    EXPECT_EQ(game1.get_current_turn(), game2.get_current_turn());
    EXPECT_EQ(game2.get_current_turn(), Player::Blue); // Red moved, so it's Blue's turn
    
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            Piece p1 = game1.get_board().get_piece(x, y);
            Piece p2 = game2.get_board().get_piece(x, y);
            EXPECT_EQ(p1.type, p2.type);
            EXPECT_EQ(p1.owner, p2.owner);
            EXPECT_EQ(p1.revealed, p2.revealed);
        }
    }

    // Clean up the test file
    std::remove(filename.c_str());
}

TEST(EngineTest, TwoSquaresRule) {
    Game game;
    game.place_piece(0, 0, PieceType::Scout, Player::Red);
    game.place_piece(9, 9, PieceType::Scout, Player::Blue);

    EXPECT_EQ(game.execute_move({0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game.execute_move({0, 1, 0, 0}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({9, 8, 9, 9}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game.execute_move({0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    
    // R attempts to return to (0,0) again, triggering the back-and-forth loop limit
    EXPECT_FALSE(game.is_legal_move({0, 1, 0, 0})); 
}

TEST(EngineTest, MoreSquaresRule) {
    Game game;
    game.place_piece(1, 1, PieceType::Scout, Player::Red);
    game.place_piece(2, 2, PieceType::Scout, Player::Blue);

    EXPECT_EQ(game.execute_move({1, 1, 1, 2}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({2, 2, 2, 3}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game.execute_move({1, 2, 2, 2}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({2, 3, 1, 3}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game.execute_move({2, 2, 2, 3}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({1, 3, 1, 2}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game.execute_move({2, 3, 1, 3}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game.execute_move({1, 2, 2, 2}), CombatResult::MovedToEmpty); // B
    
    // R chases 1,3 -> 1,2, recreating a previously visited board state hash!
    EXPECT_FALSE(game.is_legal_move({1, 3, 1, 2}));
}

TEST(EngineTest, SaveLoadHistory) {
    const std::string filename = "test_save_history.bin";
    Game game1;
    game1.place_piece(0, 0, PieceType::Scout, Player::Red);
    game1.place_piece(9, 9, PieceType::Scout, Player::Blue);

    EXPECT_EQ(game1.execute_move({0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game1.execute_move({9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game1.execute_move({0, 1, 0, 0}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game1.execute_move({9, 8, 9, 9}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(game1.execute_move({0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(game1.execute_move({9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    
    EXPECT_TRUE(game1.save_to_file(filename));

    Game game2;
    EXPECT_TRUE(game2.load_from_file(filename));
    
    // Ensures that the move history/hashes were saved and loaded correctly
    EXPECT_FALSE(game2.is_legal_move({0, 1, 0, 0}));

    std::remove(filename.c_str());
}