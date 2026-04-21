#include "../stratego/stratego.h"
#include "stratego_env.cpp"
#include <iostream>
#include <cassert>
#include <string>

using namespace stratego;

void test_channel_count() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);
    StrategoEnvironment env(board);

    StrategoObs obs = env.encode_board(board);
    int w = 4, h = 4;
    int num_playable = 4; // Flag, Major, Captain, Lieutenant
    int expected_channels = (2 * num_playable) + 3; // 11

    assert(obs.size() == expected_channels * w * h);
}

void test_chw_index() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);
    StrategoEnvironment env(board);

    assert(env.get_chw_index(0, 0, 0) == 0);
    assert(env.get_chw_index(0, 0, 1) == 1);
    assert(env.get_chw_index(0, 1, 0) == 4);
    assert(env.get_chw_index(1, 0, 0) == 16);
    assert(env.get_chw_index(2, 3, 3) == 47);
}

void test_basic_piece_placement() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);

    board.place_piece(0, 0, PieceType::Lieutenant, Player::Red);

    StrategoEnvironment env(board);
    StrategoObs obs = env.encode_board(board);

    // Lieutenant is first sorted piece type -> channel 0
    int idx_red_lt = env.get_chw_index(0, 0, 0);
    assert(obs[idx_red_lt] == 1.0f);
}

// Channels: 0-3: My Pieces, 4: Enemy Hidden, 5-8: Enemy Revealed, 9: Water, 10: Empty
// Board State (Red's Perspective):
//    0   1   2   3
// 0  .  B?  .   .   <- B? is an unrevealed Captain (Enemy Hidden, Channel 4)
// 1  .   .  .   .   <- (0,1) is Empty (Channel 10)
// 2  .   .  .   .
// 3  .   .  .   .
//
// Calculation Verification:
// - Blue Captain @ (1,0): (Channel 4 * 4 * 4) + (0 * 4) + 1 = 64 + 0 + 1 = Index 65
// - Empty Square @ (0,1): (Channel 10 * 4 * 4) + (1 * 4) + 0 = 160 + 4 + 0 = Index 164
void test_unrevealed_and_empty_channels() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);

    board.place_piece(1, 0, PieceType::Captain, Player::Blue);

    StrategoEnvironment env(board);
    StrategoObs obs = env.encode_board(board);

    int num_playable = 4;

    // Unrevealed opponent -> channel N (which is 4)
    int unrevealed_channel = num_playable;
    int idx_blue_capt = env.get_chw_index(unrevealed_channel, 0, 1);
    assert(obs[idx_blue_capt] == 1.0f);

    // Empty -> channel 2N + 2 (which is 10)
    int empty_channel = 2 * num_playable + 2;
    int idx_empty = env.get_chw_index(empty_channel, 1, 0); // (0,1) is empty
    assert(obs[idx_empty] == 1.0f);
}

// Initial board state from Red's perspective (Tiny 4x4):
// 0   R5 B6 .  .  Blue side
// 1   .  .  .  .
// 2   .  .  .  .
// 3   .  .  .  .  Red side
//     0  1  2  3
// From Red's perspective, R5 is at (0,0) and B6 is at (1,0).
// Red moves:
// 0   .  B6 .  .  Blue side
// 1   R5 .  .  .
// 2   .  .  .  .
// 3   .  .  .  .  Red side
//     0  1  2  3
// Now it is blue's turn, from their perspective R5 is at (3,3) and B6 is at (2,3).
// 0   .  .  .  .  Red side
// 1   .  .  .  .
// 2   .  .  .  R5
// 3   .  .  B6 .  Blue side
//     0  1  2  3
void test_perspective_flip() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);

    board.place_piece(0, 0, PieceType::Lieutenant, Player::Red);
    board.place_piece(1, 0, PieceType::Captain, Player::Blue);

    StrategoEnvironment env(board);

    // Turn flip
    board.execute_move({0, 0, 0, 1}); // Red moves to empty, becomes Blue's turn

    assert(board.get_current_turn() == Player::Blue);
    StrategoObs obs_blue = env.encode_board(board);
    int num_playable = 4;

    // From Blue's perspective, their Captain at (1,0) is seen at (3-1, 3-0) = (2, 3)
    // Captain is second sorted piece type -> channel 1
    int idx_blue_capt_my = env.get_chw_index(1, 3, 2);
    assert(obs_blue[idx_blue_capt_my] == 1.0f);

    // Red Lt at (0,1) is seen at (3-0, 3-1) = (3, 2).
    // It is an unrevealed opponent -> channel 4
    int unrevealed_channel = num_playable;
    int idx_red_lt_opp = env.get_chw_index(unrevealed_channel, 2, 3);
    assert(obs_blue[idx_red_lt_opp] == 1.0f);
}

//
//  --- RED'S TURN ---
// 0  R5 B6 .  .  (Blue Side)    Move (global): R5 (0,0) -> (0,1) [Relative Down]
// 1  .  .  .  .                 View: vx=0, vy=0 | Dir: 1
// 2  .  .  .  .                 Action ID: (0*4 + 0)*4 + 1 = 1
// 3  .  .  .  .  (Red Side)
//    0  1  2  3
// Resulting board:
// 0  .  B6 .  .  (Blue Side)
// 1  R5 .  .  .
// 2  .  .  .  .
// 3  .  .  .  .  (Red Side)
//    0  1  2  3
//
// --- BLUE'S TURN ----
// 0  .  .  .  .  (Red Side)     Move (global): B6 (1,0) -> (1,1) [Relative Up]
// 1  .  .  .  .                 View: vx=2, vy=3 | Dir: 0
// 2  .  .  .  R5                Action ID: (2*4 + 3)*4 + 0 = 56
// 3  .  .  B6  .  (Blue Side)
//    0  1  2  3
// Resulting board: 
// 0  .  .  .  .  (Red Side)
// 1  .  .  .  .
// 2  .  .  B6 R5
// 3  .  .  .  .  (Blue Side)
//    0  1  2  3
// ---- RED's TURN ----
// New global board state from Red's perspective:
// 0   .  .  .  .  (Blue Side)
// 1   R5 B6 .  .
// 2   .  .  .  .
// 3   .  .  .  .  (Red Side)
//    0  1  2  3
void test_action_encoding() {
    GameConfig config = get_config_for_game_type(GameType::Tiny);
    Board board(config);
    StrategoEnvironment env(board);

    // Setup initial state: Red Lt at (0,0), Blue Capt at (1,0)
    board.place_piece(0, 0, PieceType::Lieutenant, Player::Red);
    board.place_piece(1, 0, PieceType::Captain, Player::Blue);

    // --- RED'S TURN ---
    // Red move: R5 (0,0) -> (0,1) (global down, relative down)
    Move m1{0, 0, 0, 1};
    StrategoAction a1 = env.encode_action(m1);
    
    // assert(a1 == 1) // OLD
    assert(a1 == 3);   // NEW: (0*4+0)*12 + 1*3 + 0 = 3
    
    Move decoded_m1 = env.decode_action(a1);
    assert(decoded_m1.start_x == m1.start_x && decoded_m1.start_y == m1.start_y);
    assert(decoded_m1.end_x == m1.end_x && decoded_m1.end_y == m1.end_y);

    // Execute move to flip the turn to Blue
    board.execute_move(m1);
    assert(board.get_current_turn() == Player::Blue);
    env.set_board(board);

    // --- BLUE'S TURN ---
    // Blue move: B6 (1,0) -> (1,1) (global down, relative up)
    Move m2{1, 0, 1, 1};
    StrategoAction a2 = env.encode_action(m2);
    
    // assert(a2 == 56) // OLD
    assert(a2 == 168);  // NEW: (3*4+2)*12 + 0*3 + 0 = 168
    
    Move decoded_m2 = env.decode_action(a2);
    assert(decoded_m2.start_x == m2.start_x && decoded_m2.start_y == m2.start_y);
    assert(decoded_m2.end_x == m2.end_x && decoded_m2.end_y == m2.end_y);
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string test_name = argv[1];
        if (test_name == "channel_count") test_channel_count();
        else if (test_name == "chw_index") test_chw_index();
        else if (test_name == "basic_piece_placement") test_basic_piece_placement();
        else if (test_name == "unrevealed_and_empty_channels") test_unrevealed_and_empty_channels();
        else if (test_name == "perspective_flip") test_perspective_flip();
        else if (test_name == "action_encoding") test_action_encoding();
        else {
            std::cerr << "Unknown test: " << test_name << "\n";
            return 1;
        }
        return 0;
    }

    std::cout << "Running Environment Tests...\n";

    test_channel_count();
    test_chw_index();
    test_basic_piece_placement();
    test_unrevealed_and_empty_channels();
    test_perspective_flip();
    test_action_encoding();

    std::cout << "All tests passed successfully!\n";
    return 0;
}