#include <gtest/gtest.h>
#include "board.h"

using namespace stratego;

TEST(BoardTest, ClassicBoardInitializationAndPlacement) {
    Board board; // Defaults to Classic configuration
    
    // 1. Check dimensions
    EXPECT_EQ(board.get_width(), 10);
    EXPECT_EQ(board.get_height(), 10);
    
    // 2. Check lake placement (Water tiles)
    // Left lake is at x: 2,3; y: 4,5
    EXPECT_TRUE(board.get_piece(2, 4).is_obstacle());
    EXPECT_TRUE(board.get_piece(3, 5).is_obstacle());
    
    // Normal tile should be empty and not an obstacle
    EXPECT_TRUE(board.get_piece(0, 0).is_empty());
    EXPECT_FALSE(board.get_piece(0, 0).is_obstacle());
    
    // 3. Test piece placement
    // Valid placement
    EXPECT_TRUE(board.place_piece(0, 0, PieceType::Spy, Player::Red));
    EXPECT_EQ(board.get_piece(0, 0).type, PieceType::Spy);
    EXPECT_EQ(board.get_piece(0, 0).owner, Player::Red);
    EXPECT_TRUE(board.get_piece(0, 0).is_mobile()); // Spies can move
    
    // Invalid placement: On water
    EXPECT_FALSE(board.place_piece(2, 4, PieceType::Scout, Player::Blue));
    
    // Invalid placement: Out of bounds
    EXPECT_FALSE(board.place_piece(10, 10, PieceType::Flag, Player::Red));
    EXPECT_FALSE(board.place_piece(-1, 5, PieceType::Bomb, Player::Blue));
}

TEST(BoardTest, QuickBoardInitialization) {
    Board board(get_config_for_game_type(GameType::Quick));
    
    // Check dimensions
    EXPECT_EQ(board.get_width(), 8);
    EXPECT_EQ(board.get_height(), 8);
    
    // Quick lake is central: x: 2,3,4,5; y: 3,4
    EXPECT_TRUE(board.get_piece(2, 3).is_obstacle());
    EXPECT_TRUE(board.get_piece(5, 4).is_obstacle());
    EXPECT_FALSE(board.get_piece(1, 3).is_obstacle()); // Just outside the lake
}

TEST(BoardTest, ClearBoard) {
    Board board;
    board.place_piece(0, 0, PieceType::Marshal, Player::Blue);
    EXPECT_FALSE(board.get_piece(0, 0).is_empty());
    
    board.clear();
    
    // Piece should be gone, but lakes should remain
    EXPECT_TRUE(board.get_piece(0, 0).is_empty());
    EXPECT_TRUE(board.get_piece(2, 4).is_obstacle());
}