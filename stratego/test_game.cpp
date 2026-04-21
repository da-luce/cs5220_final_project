#include <gtest/gtest.h>
#include "game.h"

using namespace stratego;

TEST(GameTest, InitializationAndRandomSetup) {
    // 1. Test Default/Empty Initialization
    Game game;
    EXPECT_EQ(game.get_current_turn(), Player::Red);
    EXPECT_EQ(game.get_move_count(), 0);
    EXPECT_TRUE(game.get_history().empty());
    EXPECT_TRUE(game.get_chase_hashes().empty());
    
    // Board should be empty initially (except for water)
    EXPECT_TRUE(game.get_board().get_piece(0, 0).is_empty());

    // 2. Test Random Setup Population
    game.initialize_game(SetupType::Random);
    
    // Count pieces to ensure armies were placed correctly
    int red_count = 0;
    int blue_count = 0;
    
    const Board& board = game.get_board();
    for (int y = 0; y < board.get_height(); ++y) {
        for (int x = 0; x < board.get_width(); ++x) {
            Piece p = board.get_piece(x, y);
            if (p.owner == Player::Red) red_count++;
            if (p.owner == Player::Blue) blue_count++;
        }
    }

    // Calculate expected pieces based on the config
    int expected_pieces = 0;
    for (const auto& [type, count] : board.get_config().piece_counts) {
        expected_pieces += count;
    }

    EXPECT_EQ(red_count, expected_pieces);
    EXPECT_EQ(blue_count, expected_pieces);
}

TEST(GameTest, TurnAndHistoryManagement) {
    Game game;
    game.place_piece(0, 0, PieceType::Scout, Player::Red);
    game.place_piece(0, 9, PieceType::Scout, Player::Blue);

    // Initial state
    EXPECT_EQ(game.get_current_turn(), Player::Red);
    EXPECT_EQ(game.get_move_count(), 0);

    // Make a move
    CombatResult res1 = game.execute_move({0, 0, 0, 1});
    EXPECT_EQ(res1, CombatResult::MovedToEmpty);

    // State should update correctly
    EXPECT_EQ(game.get_current_turn(), Player::Blue);
    EXPECT_EQ(game.get_move_count(), 1);
    EXPECT_EQ(game.get_history().size(), 1);
    EXPECT_EQ(game.get_chase_hashes().size(), 1);
    
    // Move history should exactly match what was played
    Move played = game.get_history()[0];
    EXPECT_EQ(played.start_x, 0);
    EXPECT_EQ(played.start_y, 0);
    EXPECT_EQ(played.end_x, 0);
    EXPECT_EQ(played.end_y, 1);
}

TEST(GameTest, MaxMovesDrawCondition) {
    // Create a custom config with a strict move limit
    BoardConfig config = get_config_for_game_type(GameType::Classic);
    int max_moves = 2; // Match ends after 2 total moves
    
    Game game(config, max_moves);
    game.place_piece(0, 0, PieceType::Scout, Player::Red);
    game.place_piece(9, 9, PieceType::Scout, Player::Blue);

    // Move 1 (Red)
    EXPECT_EQ(game.execute_move({0, 0, 0, 1}), CombatResult::MovedToEmpty);
    EXPECT_EQ(game.get_move_count(), 1);

    // Move 2 (Blue) - Hits the move limit!
    // Even though the move is mechanically just "MovedToEmpty", the Game wrapper
    // intercepts it and declares a Draw.
    EXPECT_EQ(game.execute_move({9, 9, 9, 8}), CombatResult::Draw);
    EXPECT_EQ(game.get_move_count(), 2);
}

TEST(GameTest, HashConsistency) {
    Game game;
    game.place_piece(0, 0, PieceType::Miner, Player::Red);
    
    GameHash initial_hash = game.get_current_hash();
    
    // Moving the piece changes the hash
    game.execute_move({0, 0, 0, 1});
    GameHash step_2_hash = game.get_current_hash();
    EXPECT_NE(initial_hash, step_2_hash);

    // A completely different game with the same layout should hash the same
    Game game2;
    game2.place_piece(0, 0, PieceType::Miner, Player::Red);
    EXPECT_EQ(game2.get_current_hash(), initial_hash);
}