#pragma once
#include "../stratego/stratego.h"
#include <string>

namespace stratego {

struct GameState {
    int cursor_x = 0;
    int cursor_y = 0;
    int selected_x = -1;
    int selected_y = -1;
    bool exit_game = false;
    bool game_over = false;
    bool in_confrontation = false;
    bool casual_mode = false;
    Move pending_move{};
    PieceType pending_attacker_type{PieceType::Empty};
    PieceType pending_defender_type{PieceType::Empty};
    int last_ch = -1;
    std::string status_msg = "Welcome! You are RED. Use Arrows to move, ENTER to select.";
    std::string ai_type = "Random AI";
    std::string model_path = "";
};

// String/UI helpers
const char* piece_to_str(PieceType type);
const char* piece_name(PieceType type);

bool show_start_menu(Board& board, GameState& state);
void render_board(const Board& board, const GameState& state);

} // namespace stratego