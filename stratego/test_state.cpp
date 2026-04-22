#include <gtest/gtest.h>
#include "state.h"
#include "engine.h"

using namespace stratego;

TEST(StateTest, InitializationAndRandomSetup) {
    // 1. Test Default/Empty Initialization
    GameState state;
    EXPECT_EQ(state.current_turn, Player::Red);
    EXPECT_EQ(state.move_count, 0);
    EXPECT_TRUE(state.move_history.empty());
    EXPECT_TRUE(state.chase_hashes.empty());
    
    // Board should be empty initially (except for water)
    EXPECT_TRUE(state.board.get_piece(0, 0).is_empty());

    // 2. Test Random Setup Population
    state = state::initialize(get_config_for_game_type(GameType::Classic), state::SetupType::Random, 2000);
    
    // Count pieces to ensure armies were placed correctly
    int red_count = 0;
    int blue_count = 0;
    
    const Board& board = state.board;
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

TEST(StateTest, SerializeDeserializeHistory) {
    GameState state1;
    state1.board.place_piece(0, 0, PieceType::Scout, Player::Red);
    state1.board.place_piece(9, 9, PieceType::Scout, Player::Blue);

    EXPECT_EQ(Engine::execute_move(state1, {0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state1, {9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state1, {0, 1, 0, 0}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state1, {9, 8, 9, 9}), CombatResult::MovedToEmpty); // B
    EXPECT_EQ(Engine::execute_move(state1, {0, 0, 0, 1}), CombatResult::MovedToEmpty); // R
    EXPECT_EQ(Engine::execute_move(state1, {9, 9, 9, 8}), CombatResult::MovedToEmpty); // B
    
    state::GameBinary data = state::serialize(state1);

    GameState state2;
    EXPECT_NO_THROW(state2 = state::deserialize(data));
    
    // Ensures that the move history/hashes were saved and loaded correctly
    EXPECT_FALSE(Engine::is_legal_move(state2, {0, 1, 0, 0}));
}

TEST(StateTest, HashConsistency) {
    GameState state;
    state.board.place_piece(0, 0, PieceType::Miner, Player::Red);
    
    GameHash initial_hash = state.board.compute_hash(state.current_turn);
    
    // Moving the piece changes the hash
    Engine::execute_move(state, {0, 0, 0, 1});
    GameHash step_2_hash = state.board.compute_hash(state.current_turn);
    EXPECT_NE(initial_hash, step_2_hash);

    // A completely different game with the same layout should hash the same
    GameState state2;
    state2.board.place_piece(0, 0, PieceType::Miner, Player::Red);
    EXPECT_EQ(state2.board.compute_hash(state2.current_turn), initial_hash);
}