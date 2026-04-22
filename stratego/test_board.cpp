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

TEST(BoardTest, PlayerViewMasking) {
    Board board;
    
    // Setup: Red piece, Blue piece (unrevealed), and Blue piece (revealed)
    board.place_piece(0, 0, PieceType::Marshal, Player::Red);
    board.place_piece(1, 1, PieceType::Spy, Player::Blue);
    board.place_piece(2, 2, PieceType::General, Player::Blue);
    
    // Manually reveal the Blue General
    int gen_idx = board.index(2, 2);
    board.grid[gen_idx].revealed = true;

    // Get Red's view
    Board red_view = board.get_player_view(Player::Red);

    // 1. Red should see their own piece normally
    EXPECT_EQ(red_view.get_piece(0, 0).type, PieceType::Marshal);
    EXPECT_EQ(red_view.get_piece(0, 0).owner, Player::Red);

    // 2. Red should see the unrevealed Blue Spy as "Hidden"
    EXPECT_EQ(red_view.get_piece(1, 1).type, PieceType::Hidden);
    EXPECT_EQ(red_view.get_piece(1, 1).owner, Player::Blue);

    // 3. Red should see the revealed Blue General normally
    EXPECT_EQ(red_view.get_piece(2, 2).type, PieceType::General);
    EXPECT_EQ(red_view.get_piece(2, 2).owner, Player::Blue);
    
    // 4. Water should remain water
    EXPECT_TRUE(red_view.get_piece(2, 4).is_obstacle());
}

TEST(BoardTest, FlippedBoardTransformation) {
    Board board;
    int w = board.get_width();
    int h = board.get_height();

    // Place a piece in the top-left corner (0, 0)
    board.place_piece(0, 0, PieceType::Marshal, Player::Red);
    // Place a piece in a non-symmetrical spot (1, 2)
    board.place_piece(1, 2, PieceType::Spy, Player::Blue);

    Board flipped = board.get_flipped_board();

    // 1. (0, 0) should move to (w-1, h-1) -> (9, 9) in classic
    EXPECT_EQ(flipped.get_piece(w - 1, h - 1).type, PieceType::Marshal);
    EXPECT_EQ(flipped.get_piece(w - 1, h - 1).owner, Player::Red);

    // 2. (1, 2) should move to (w-1-1, h-1-2) -> (8, 7) in classic
    EXPECT_EQ(flipped.get_piece(w - 2, h - 3).type, PieceType::Spy);
    EXPECT_EQ(flipped.get_piece(w - 2, h - 3).owner, Player::Blue);

    // 3. Verify the original spots in the flipped board are now empty
    EXPECT_TRUE(flipped.get_piece(0, 0).is_empty());
    
    // 4. Verify that flipping a flipped board gets you back to the original
    Board double_flipped = flipped.get_flipped_board();
    EXPECT_EQ(double_flipped.get_piece(0, 0).type, PieceType::Marshal);
    EXPECT_EQ(double_flipped.get_piece(1, 2).type, PieceType::Spy);
}