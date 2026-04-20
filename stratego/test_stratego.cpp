#include "stratego.h"
#include <iostream>
#include <cassert>
#include <cstdio> // For std::remove

using namespace stratego;

void test_out_of_bounds() {
    Board board;
    board.place_piece(0, 0, PieceType::Sergeant, Player::Red);

    // Illegal moves outside the board dimensions
    assert(!board.is_legal_move({0, 0, -1, 0}));
    assert(!board.is_legal_move({0, 0, 0, -1}));
    assert(!board.is_legal_move({0, 0, 10, 0}));
    assert(!board.is_legal_move({0, 0, 0, 10}));
}

void test_immobile_pieces() {
    Board board;
    board.place_piece(0, 0, PieceType::Flag, Player::Red);
    board.place_piece(1, 0, PieceType::Bomb, Player::Red);

    // Flags and Bombs cannot be moved
    assert(!board.is_legal_move({0, 0, 0, 1}));
    assert(!board.is_legal_move({1, 0, 1, 1}));
}

void test_standard_moves() {
    Board board;
    board.place_piece(1, 1, PieceType::Captain, Player::Red);

    // Standard pieces can move exactly 1 tile orthogonally
    assert(board.is_legal_move({1, 1, 1, 2}));
    assert(board.is_legal_move({1, 1, 2, 1}));

    // Standard pieces cannot move > 1 tile
    assert(!board.is_legal_move({1, 1, 1, 3}));
    
    // Standard pieces cannot move diagonally
    assert(!board.is_legal_move({1, 1, 2, 2}));
}

void test_scout_moves() {
    Board board;
    
    // In a standard 10x10, water is at x in [2,3] and [6,7], y in [4,5]
    board.place_piece(2, 2, PieceType::Scout, Player::Red);

    // 1. Moving across water is illegal
    assert(!board.is_legal_move({2, 2, 2, 6})); 
    
    // Moving INTO water is also illegal
    assert(!board.is_legal_move({2, 2, 2, 4}));

    // 2. Crossing multiple enemy pieces is illegal
    board.place_piece(0, 0, PieceType::Scout, Player::Red);
    board.place_piece(0, 1, PieceType::Sergeant, Player::Blue);
    board.place_piece(0, 2, PieceType::Sergeant, Player::Blue);
    
    // Attempt to move past them to an empty square
    assert(!board.is_legal_move({0, 0, 0, 3}));
    // Attempt to attack the second one by jumping the first
    assert(!board.is_legal_move({0, 0, 0, 2}));

    // 3. Crossing your own pieces is illegal
    board.place_piece(9, 0, PieceType::Scout, Player::Red);
    board.place_piece(9, 1, PieceType::Spy, Player::Red);
    assert(!board.is_legal_move({9, 0, 9, 2}));

    // 4. Valid long-distance attack
    board.place_piece(5, 0, PieceType::Scout, Player::Red);
    board.place_piece(5, 3, PieceType::Major, Player::Blue);
    assert(board.is_legal_move({5, 0, 5, 3})); // Legal attack
}

void test_combat_resolutions() {
    Board board;
    board.place_piece(0, 0, PieceType::Miner, Player::Red);
    board.place_piece(0, 1, PieceType::Bomb, Player::Blue);
    
    assert(board.is_legal_move({0, 0, 0, 1}));
    CombatResult res1 = board.execute_move({0, 0, 0, 1});
    assert(res1 == CombatResult::AttackerWins); // Miner defuses bomb

    // Next turn is Blue's
    board.place_piece(9, 9, PieceType::Spy, Player::Blue);
    board.place_piece(9, 8, PieceType::Marshal, Player::Red);
    
    CombatResult res2 = board.execute_move({9, 9, 9, 8});
    assert(res2 == CombatResult::AttackerWins); // Spy kills Marshal
}

void test_save_load() {
    const std::string filename = "test_save.bin";
    
    Board board1;
    board1.place_piece(3, 7, PieceType::Marshal, Player::Red);
    board1.place_piece(3, 6, PieceType::Spy, Player::Red); // Attacker
    board1.place_piece(3, 5, PieceType::General, Player::Blue); // Defender
    
    // Make a move to change turn and reveal a piece.
    // Red Spy (1) attacks Blue General (9). General wins, Spy is removed.
    board1.execute_move({3, 6, 3, 5}); 
    
    // Save the state after the move
    assert(board1.save_to_file(filename));

    // Load the state into a new, empty board
    Board board2;
    assert(board2.load_from_file(filename));

    // Compare the two boards to ensure they are identical
    assert(board1.get_current_turn() == board2.get_current_turn());
    assert(board2.get_current_turn() == Player::Blue); // Red moved, so it's Blue's turn
    
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            Piece p1 = board1.get_piece(x, y);
            Piece p2 = board2.get_piece(x, y);
            assert(p1.type == p2.type);
            assert(p1.owner == p2.owner);
            assert(p1.revealed == p2.revealed);
        }
    }

    // Clean up the test file
    std::remove(filename.c_str());
}

int main() {
    std::cout << "Running Stratego Rules Tests...\n";
    
    test_out_of_bounds();
    test_immobile_pieces();
    test_standard_moves();
    test_scout_moves();
    test_combat_resolutions();
    test_save_load();

    std::cout << "All tests passed successfully!\n";
    return 0;
}