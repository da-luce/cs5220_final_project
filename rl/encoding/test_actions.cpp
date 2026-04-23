#include <gtest/gtest.h>

#include "actions.h"
#include "stratego/board.h"
#include "stratego/state.h"

using encoding::actions::ActionEncoder;

namespace {

// Helper to build a simple board config
stratego::BoardConfig make_config(int w, int h) {
    stratego::BoardConfig config;
    config.width = w;
    config.height = h;
    return config;
}

// Helper to create a move
stratego::Move make_move(int fx, int fy, int tx, int ty) {
    stratego::Move m;
    m.start_x = fx;
    m.start_y = fy;
    m.end_x = tx;
    m.end_y = ty;
    return m;
}

} // namespace

// ----------------------------
// Basic initialization tests
// ----------------------------

TEST(ActionEncoderTest, CorrectChannelCount) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    // D_max = max(10,10) - 1 = 9
    // total_channels = 4 * 9 = 36
    EXPECT_EQ(encoder.get_action_channels(), 36);
}

TEST(ActionEncoderTest, WorksForRectangularBoard) {
    auto config = make_config(8, 5);
    ActionEncoder encoder(config);

    // D_max = max(8,5) - 1 = 7
    EXPECT_EQ(encoder.get_action_channels(), 28);
}

// ----------------------------
// Move → channel tests
// ----------------------------

TEST(ActionEncoderTest, EncodeNorthMove) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 4, 3); // North 1
    int channel = encoder.move_to_channel(move);

    EXPECT_EQ(channel, 0); // North distance 1 → channel 0
}

TEST(ActionEncoderTest, EncodeSouthMove) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 4, 6); // South 2
    int channel = encoder.move_to_channel(move);

    // South starts at D_max = 9
    EXPECT_EQ(channel, 9 + 1); // distance 2 → index 1
}

TEST(ActionEncoderTest, EncodeEastMove) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 7, 4); // East 3
    int channel = encoder.move_to_channel(move);

    EXPECT_EQ(channel, 18 + 2); // East block starts at 2*9
}

TEST(ActionEncoderTest, EncodeWestMove) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 2, 4); // West 2
    int channel = encoder.move_to_channel(move);

    EXPECT_EQ(channel, 27 + 1); // West block starts at 3*9
}

// ----------------------------
// Round-trip tests
// ----------------------------

TEST(ActionEncoderTest, RoundTripNorth) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(5, 5, 5, 2); // North 3

    int channel = encoder.move_to_channel(move);
    auto decoded = encoder.channel_to_move(channel, 5, 5);

    EXPECT_EQ(decoded.end_x, move.end_x);
    EXPECT_EQ(decoded.end_y, move.end_y);
}

TEST(ActionEncoderTest, RoundTripAllDirections) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    std::vector<stratego::Move> moves = {
        make_move(5, 5, 5, 3), // North 2
        make_move(5, 5, 5, 8), // South 3
        make_move(5, 5, 8, 5), // East 3
        make_move(5, 5, 2, 5)  // West 3
    };

    for (const auto& move : moves) {
        int channel = encoder.move_to_channel(move);
        auto decoded = encoder.channel_to_move(channel, move.start_x, move.start_y);

        EXPECT_EQ(decoded.end_x, move.end_x);
        EXPECT_EQ(decoded.end_y, move.end_y);
    }
}

// ----------------------------
// Edge cases
// ----------------------------

TEST(ActionEncoderTest, MaxDistanceMove) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(0, 9, 0, 0); // North 9 (max)

    int channel = encoder.move_to_channel(move);

    EXPECT_EQ(channel, 8); // last North channel (9-1)
}

TEST(ActionEncoderTest, InvalidDiagonalMoveThrows) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 5, 5); // diagonal

    EXPECT_THROW(encoder.move_to_channel(move), std::invalid_argument);
}

TEST(ActionEncoderTest, InvalidZeroMoveThrows) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    auto move = make_move(4, 4, 4, 4); // no movement

    EXPECT_THROW(encoder.move_to_channel(move), std::invalid_argument);
}

TEST(ActionEncoderTest, ChannelOutOfRangeThrows) {
    auto config = make_config(10, 10);
    ActionEncoder encoder(config);

    EXPECT_THROW(encoder.channel_to_move(100, 4, 4), std::out_of_range);
}