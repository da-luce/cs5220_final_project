#pragma once
#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../stratego/policy.h"
#include <string>
#include <optional>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

namespace stratego {

enum class AIType {
    Random,
    Trained
};

struct UIGameState {
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
    std::string last_input = "";
    std::string status_msg = "Welcome! You are RED. Use Arrows to move, ENTER to select.";
    std::string ai_type = "Random AI";
    std::string model_path = "";
};

// String/UI helpers
const char* piece_to_str(PieceType type);
const char* piece_name(PieceType type);

struct GameSettings {
    GameType game_type;
    state::SetupType setup_type;
    AIType ai_type;
    bool new_game = true;
    std::string load_filename;
    std::string model_path;
};

std::optional<GameSettings> start_menu();

ftxui::Element render_board(const GameState& game_state, const UIGameState& state);

} // namespace stratego