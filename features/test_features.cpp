#include <gtest/gtest.h>
#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "features.h"

using namespace stratego;

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

// Helper to calculate 1D index for flattened CHW vector
inline int get_chw_index(int c, int y, int x, int width, int height) {
    return (c * height * width) + (y * width) + x;
}

// Helper to determine what dynamic channel a piece was assigned to
inline int get_dynamic_channel(const BoardConfig& config, PieceType target_type) {
    int channel = 0;
    for (const auto& [ptype, count] : config.piece_counts) {
        if (ptype == target_type) return channel;
        channel++;
    }
    return -1; // Not found
}

// -----------------------------------------------------------------------------
// Test Fixture
// -----------------------------------------------------------------------------
class FeatureEncodingTest : public ::testing::Test {
protected:
    BoardConfig tiny_config;
    int w, h;
    int num_playable;
    int total_channels;

    void SetUp() override {
        tiny_config = get_config_for_game_type(GameType::Tiny);
        w = tiny_config.width;   
        h = tiny_config.height;  
        num_playable = tiny_config.piece_counts.size(); // E.g., 4 for Tiny
        total_channels = (2 * num_playable) + 2;        // E.g., 10 for Tiny
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_F(FeatureEncodingTest, DynamicChannelCount) {
    GameState state;
    state.board = Board(tiny_config);
    
    BoardFeatures obs = get_board_encoding(state, Player::Red);
    
    // Check that the vector length exactly matches the expected flattened CHW volume
    EXPECT_EQ(obs.size(), static_cast<size_t>(total_channels * w * h));
}

TEST_F(FeatureEncodingTest, BasicPiecePlacement) {
    GameState state;
    state.board = Board(tiny_config);
    
    state.board.place_piece(0, 0, PieceType::Lieutenant, Player::Red);

    BoardFeatures obs = get_board_encoding(state, Player::Red);

    // Determine which dynamic channel Lieutenant mapped to
    int target_channel = get_dynamic_channel(tiny_config, PieceType::Lieutenant);
    ASSERT_NE(target_channel, -1) << "Lieutenant must be part of Tiny config for this test.";

    int idx_red_lt = get_chw_index(target_channel, 0, 0, w, h);
    
    EXPECT_FLOAT_EQ(obs[idx_red_lt], 1.0f);
}

/* * Channels: 0 to N-1: My Pieces, N: Enemy Hidden, N+1 to 2N: Enemy Revealed, 2N+1: Water
 * Board State (Red's Perspective):
 * 0   1   2   3
 * 0  .  B?   .   .   <- B? is an unrevealed Captain (Enemy Hidden, Channel N)
 * 1  .   .   .   .   <- (0,1) is Empty (Implicitly 0 across all channels)
 * 2  .   .   .   .
 * 3  .   .   .   .
 */
TEST_F(FeatureEncodingTest, UnrevealedAndEmptyChannels) {
    GameState state;
    state.board = Board(tiny_config);

    // Place an unrevealed Blue Captain
    state.board.place_piece(1, 0, PieceType::Captain, Player::Blue);

    BoardFeatures obs = get_board_encoding(state, Player::Red);

    // Unrevealed opponent -> channel N
    int unrevealed_channel = num_playable;
    int idx_blue_capt = get_chw_index(unrevealed_channel, 0, 1, w, h);
    
    EXPECT_FLOAT_EQ(obs[idx_blue_capt], 1.0f);

    // Ensure empty square (0,1) is zeroed out in the unrevealed channel
    int idx_empty = get_chw_index(unrevealed_channel, 1, 0, w, h); 
    EXPECT_FLOAT_EQ(obs[idx_empty], 0.0f);
}

/*
 * Initial board state from Red's perspective (Tiny 4x4):
 * 0   R5 B6 .  .  Blue side
 * 1   .  .  .  .
 * 2   .  .  .  .
 * 3   .  .  .  .  Red side
 * 0  1  2  3
 * Red moves R5 (0,0) to (0,1). Engine flips turn to Blue.
 * Now it is blue's turn, from their perspective R5 is at (3,2) and B6 is at (2,3).
 * 0   .  .  .  .  Red side
 * 1   .  .  .  .
 * 2   .  .  .  R5
 * 3   .  .  B6 .  Blue side
 * 0  1  2  3
 */
TEST_F(FeatureEncodingTest, PerspectiveFlip) {
    GameState state;
    state.board = Board(tiny_config);
    state.current_turn = Player::Red;

    state.board.place_piece(0, 0, PieceType::Lieutenant, Player::Red);
    state.board.place_piece(1, 0, PieceType::Captain, Player::Blue);

    // Execute Move via Engine (Red moving down to empty space)
    Move m{0, 0, 0, 1};
    Engine::execute_move(state, m);

    ASSERT_EQ(state.current_turn, Player::Blue);
    
    // Generate observation from BLUE's perspective
    BoardFeatures obs_blue = get_board_encoding(state, Player::Blue);

    int capt_base_channel = get_dynamic_channel(tiny_config, PieceType::Captain);
    ASSERT_NE(capt_base_channel, -1);

    // From Blue's perspective, their Captain at global (1,0) is seen at (3-1, 3-0) = (2, 3)
    int idx_blue_capt_my = get_chw_index(capt_base_channel, 3, 2, w, h);
    EXPECT_FLOAT_EQ(obs_blue[idx_blue_capt_my], 1.0f);

    // Red Lt at global (0,1) is seen at (3-0, 3-1) = (3, 2).
    // It is an unrevealed opponent -> channel N
    int unrevealed_channel = num_playable;
    int idx_red_lt_opp = get_chw_index(unrevealed_channel, 2, 3, w, h);
    EXPECT_FLOAT_EQ(obs_blue[idx_red_lt_opp], 1.0f);
}

// Ensure the feature extractor crashes safely if a rogue piece is placed
TEST_F(FeatureEncodingTest, DynamicConfigGuardrail) {
    GameState state;
    state.board = Board(tiny_config);
    
    // Place a Marshal (Type 10). A Marshal does not exist in the Tiny BoardConfig!
    state.board.place_piece(0, 0, PieceType::Marshal, Player::Red);

    // The feature encoder should detect that Marshal is missing from piece_to_base_channel
    EXPECT_THROW({
        get_board_encoding(state, Player::Red);
    }, std::runtime_error);
}

// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "Running Data-Oriented Feature Extractor Tests...\n";
    return RUN_ALL_TESTS();
}