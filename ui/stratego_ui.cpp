#include "stratego_ui.h"
#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <filesystem>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

namespace stratego {

const char* piece_to_str(PieceType type) {
    switch (type) {
        case PieceType::Spy: return "S";
        case PieceType::Scout: return "2";
        case PieceType::Miner: return "3";
        case PieceType::Sergeant: return "4";
        case PieceType::Lieutenant: return "5";
        case PieceType::Captain: return "6";
        case PieceType::Major: return "7";
        case PieceType::Colonel: return "8";
        case PieceType::General: return "9";
        case PieceType::Marshal: return "★"; // Star for Marshal
        case PieceType::Bomb: return "B";
        case PieceType::Flag: return "⚑";    // Unicode Flag
        case PieceType::Water: return "≈≈";   // Water Waves
        default: return "·";
    }
}

const char* piece_name(PieceType type) {
    switch (type) {
        case PieceType::Spy: return "Spy";
        case PieceType::Scout: return "Scout";
        case PieceType::Miner: return "Miner";
        case PieceType::Sergeant: return "Sergeant";
        case PieceType::Lieutenant: return "Lieutenant";
        case PieceType::Captain: return "Captain";
        case PieceType::Major: return "Major";
        case PieceType::Colonel: return "Colonel";
        case PieceType::General: return "General";
        case PieceType::Marshal: return "Marshal";
        case PieceType::Bomb: return "Bomb";
        case PieceType::Flag: return "Flag";
        default: return "Piece";
    }
}

std::optional<GameSettings> start_menu() {
    using namespace ftxui;
    auto screen = ScreenInteractive::Fullscreen();
    GameSettings settings;
    bool quit = false;
    bool finished = false;

    int step = 0; // 0: Game Type, 1: Layout, 2: AI, 3: Model, 4: Done
    int tab_index = 0; // 0: game_type, 1: load_input, 2: layout, 3: ai, 4: model
    std::string error_msg = "";

    MenuOption option;
    option.entries_option.transform = [](const EntryState& state) {
        Element e = text(state.label);
        if (state.active) {
            return hbox({text("  > ") | color(Color::Cyan), e | color(Color::Cyan)});
        }
        return hbox({text("    "), e});
    };

    // Game Type
    int game_type_selected = 0;
    std::vector<std::string> game_type_entries = {
        "Classic (10x10)", "Barrage (10x10)", "Quick (8x8)", "Tiny (4x4)", "Load Game"
    };
    auto game_type_menu = Menu(&game_type_entries, &game_type_selected, option);

    // Layout
    int layout_selected = 0;
    std::vector<std::string> layout_entries = {"Probabilistic", "Random"};
    auto layout_menu = Menu(&layout_entries, &layout_selected, option);

    // AI Type
    int ai_selected = 0;
    std::vector<std::string> ai_entries = {"Random AI", "Trained AI"};
    auto ai_menu = Menu(&ai_entries, &ai_selected, option);

    // Load filename
    std::string load_filename = "";
    auto load_input = Input(&load_filename, "");

    // Model selection
    int model_selected = 0;
    std::vector<std::string> model_entries;
    auto model_menu = Menu(&model_entries, &model_selected, option);

    auto container = Container::Tab({
        game_type_menu,
        load_input,
        layout_menu,
        ai_menu,
        model_menu
    }, &tab_index);

    auto main_renderer = Renderer(container, [&] {
        Elements lines;

        auto format_question = [&](const std::string& q, bool is_active, const std::string& answer = "") {
            if (is_active) {
                return hbox({text("? ") | color(Color::Cyan) | bold, text(q) | bold});
            } else {
                return hbox({
                    text("✔ ") | color(Color::Green) | bold,
                    text(q) | bold,
                    text(" … ") | dim,
                    filler(),
                    text(answer) | color(Color::Cyan)
                });
            }
        };

        lines.push_back(text(""));

        if (step >= 0) {
            lines.push_back(format_question("Which Game Type would you like to play?", step == 0, game_type_entries[game_type_selected]));
            if (step == 0) lines.push_back(game_type_menu->Render());
        }

        if (step >= 1 && game_type_entries[game_type_selected] == "Load Game") {
            lines.push_back(format_question("Enter filename", step == 1, load_filename));
            if (step == 1) lines.push_back(hbox({text("  > ") | color(Color::Cyan), load_input->Render() | color(Color::Cyan)}));
        }

        if (step >= 1 && game_type_entries[game_type_selected] != "Load Game") {
            lines.push_back(format_question("Select Starting Layout", step == 1, layout_entries[layout_selected]));
            if (step == 1) lines.push_back(layout_menu->Render());
        }

        if (step >= 2 && game_type_entries[game_type_selected] != "Load Game") {
            lines.push_back(format_question("Select AI Opponent", step == 2, ai_entries[ai_selected]));
            if (step == 2) lines.push_back(ai_menu->Render());
        }

        if (step >= 3 && game_type_entries[game_type_selected] != "Load Game" && ai_entries[ai_selected] == "Trained AI") {
            lines.push_back(format_question("Select Model", step == 3, model_entries.empty() ? "" : model_entries[model_selected]));
            
            std::string variant_cmd = (settings.game_type == GameType::Classic) ? "classic" :
                                      (settings.game_type == GameType::Barrage) ? "barrage" :
                                      (settings.game_type == GameType::Quick) ? "quick" : "tiny";
            std::string setup_cmd = (settings.setup_type == state::SetupType::Probabilistic) ? "probabilistic" :
                                    (settings.setup_type == state::SetupType::Default) ? "default" : "random";
            std::string train_cmd = "./build/rl/train " + variant_cmd + " " + setup_cmd + " 20000";

            if (step == 3) {
                if (model_entries.empty()) {
                    lines.push_back(text("    No models found!") | color(Color::Red));
                    lines.push_back(text("    " + error_msg) | color(Color::Red));
                    lines.push_back(text(""));
                    lines.push_back(text("    To train a model for this config, run:") | dim);
                    lines.push_back(text("    " + train_cmd) | color(Color::Yellow));
                } else {
                    lines.push_back(model_menu->Render());
                    lines.push_back(text(""));
                    lines.push_back(text("    To train a new model, run:") | dim);
                    lines.push_back(text("    " + train_cmd) | color(Color::Yellow));
                }
            }
        }

        lines.push_back(text(""));
        lines.push_back(text("  Q to quit • ESC to go back") | dim);

        return vbox(lines) | size(WIDTH, EQUAL, 65);
    });

    auto catch_events = CatchEvent(main_renderer, [&](Event event) {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            quit = true;
            screen.Exit();
            return true;
        }
        if (event == Event::Escape) {
            if (step > 0) {
                step--;
                if (step == 0) tab_index = 0;
                else if (step == 1) {
                    if (game_type_entries[game_type_selected] == "Load Game") tab_index = 1;
                    else tab_index = 2;
                }
                else if (step == 2) tab_index = 3;
                return true;
            } else {
                quit = true;
                screen.Exit();
                return true;
            }
        }
        if (event == Event::Return) {
            if (step == 0) {
                if (game_type_entries[game_type_selected] == "Load Game") {
                    step = 1;
                    tab_index = 1;
                } else {
                    settings.new_game = true;
                    settings.game_type = (game_type_selected == 0) ? GameType::Classic :
                                         (game_type_selected == 1) ? GameType::Barrage :
                                         (game_type_selected == 2) ? GameType::Quick : GameType::Tiny;
                    step = 1;
                    tab_index = 2;
                }
                return true;
            }
            if (step == 1) {
                if (game_type_entries[game_type_selected] == "Load Game") {
                    if (!load_filename.empty()) {
                        settings.new_game = false;
                        settings.load_filename = load_filename;
                        finished = true;
                        screen.Exit();
                    }
                    return true;
                } else {
                    settings.setup_type = (layout_selected == 0) ? state::SetupType::Probabilistic : state::SetupType::Random;
                    step = 2;
                    tab_index = 3;
                    return true;
                }
            }
            if (step == 2) {
                settings.ai_type = (ai_selected == 1) ? AIType::Trained : AIType::Random;
                if (settings.ai_type == AIType::Random) {
                    finished = true;
                    screen.Exit();
                } else {
                    // Populate models
                    model_entries.clear();
                    std::string variant_str = (settings.game_type == GameType::Classic) ? "classic" :
                                              (settings.game_type == GameType::Barrage) ? "barrage" :
                                              (settings.game_type == GameType::Quick) ? "quick" : "tiny";
                    std::string setup_str = (settings.setup_type == state::SetupType::Probabilistic) ? "probabilistic" :
                                            (settings.setup_type == state::SetupType::Default) ? "default" : "random";
                    std::string model_prefix = variant_str + "_" + setup_str;
                    
                    std::filesystem::path models_path = std::filesystem::path(PROJECT_ROOT_DIR) / "models";
                    if (std::filesystem::exists(models_path)) {
                        for (const auto& entry : std::filesystem::directory_iterator(models_path)) {
                            if (entry.is_regular_file() && entry.path().extension() == ".pt") {
                                std::string filename = entry.path().filename().string();
                                if (filename == model_prefix + ".pt") {
                                    model_entries.push_back(filename);
                                }
                            }
                        }
                    }
                    
                    if (model_entries.empty()) {
                        error_msg = "No models found matching: " + model_prefix + ".pt";
                    }
                    
                    step = 3;
                    tab_index = 4;
                }
                return true;
            }
            if (step == 3) {
                if (!model_entries.empty()) {
                    std::filesystem::path full_path = std::filesystem::path(PROJECT_ROOT_DIR) / "models" / model_entries[model_selected];
                    settings.model_path = full_path.string();
                    finished = true;
                    screen.Exit();
                } else {
                    step = 0; // Reset or back
                    tab_index = 0;
                }
                return true;
            }
        }
        return false;
    });

    screen.Loop(catch_events);

    if (quit || !finished) return std::nullopt;
    return settings;
}

ftxui::Element render_board(const GameState& game_state, const UIGameState& state) {
    using namespace ftxui;
    
    const Board& board = game_state.board;
    int w = board.get_width();
    int h = board.get_height();

    std::vector<Move> active_legal_moves;
    if (state.selected_x != -1 && state.selected_y != -1) {
        active_legal_moves = Engine::get_legal_moves_for_piece(game_state, state.selected_x, state.selected_y);
    }

    Elements grid_rows;
    
    // Header
    Elements header_row;
    header_row.push_back(text("  ")); // offset for y labels
    for (int x = 0; x < w; ++x) {
        header_row.push_back(text(std::to_string(x)) | size(WIDTH, EQUAL, 3) | center);
    }
    grid_rows.push_back(hbox(header_row));

    for (int y = 0; y < h; ++y) {
        Elements row_elements;
        row_elements.push_back(text(std::to_string(y) + " "));

        for (int x = 0; x < w; ++x) {
            Piece p = board.get_piece(x, y);
            bool is_cursor = (x == state.cursor_x && y == state.cursor_y) && !state.game_over;
            bool is_selected = (x == state.selected_x && y == state.selected_y);

            bool is_valid_empty = false;
            bool is_valid_attack = false;
            for (const auto& m : active_legal_moves) {
                if (m.end_x == x && m.end_y == y) {
                    if (p.is_empty()) is_valid_empty = true;
                    else is_valid_attack = true;
                    break;
                }
            }

            bool is_confrontation_target = (state.in_confrontation && x == state.pending_move.end_x && y == state.pending_move.end_y);
            bool is_confrontation_source = (state.in_confrontation && x == state.pending_move.start_x && y == state.pending_move.start_y);

            Element cell;

            if (is_confrontation_source) {
                cell = text("·");
            } else if (is_confrontation_target) {
                std::string a_str = piece_to_str(state.pending_attacker_type);
                std::string d_str = piece_to_str(state.pending_defender_type);
                
                Player current_player = game_state.current_turn;
                Color attacker_color = (current_player == Player::Red) ? Color::Red : Color::Blue;
                Color defender_color = (current_player == Player::Red) ? Color::Blue : Color::Red;

                cell = hbox(
                    text(a_str) | color(attacker_color),
                    text(d_str) | color(defender_color)
                );
            }
            else if (p.is_obstacle()) cell = text("≈≈") | color(Color::Cyan);
            else if (p.is_empty()) cell = text("·");
            else if (p.owner == Player::Red) cell = text(std::string(piece_to_str(p.type))) | color(Color::Red);
            else {
                if (state.game_over || (state.casual_mode && p.revealed)) {
                    cell = text(std::string(piece_to_str(p.type))) | color(Color::Blue);
                } else {
                    cell = text("?") | color(Color::Magenta);
                }
            }

            // Apply background colors for cursor/selection
            if (is_selected) cell = cell | bgcolor(Color::Yellow) | color(Color::Black);
            else if (is_cursor) cell = cell | bgcolor(Color::White) | color(Color::Black);
            else if (is_valid_attack) cell = cell | bgcolor(Color::Red) | color(Color::White);
            else if (is_valid_empty) cell = cell | bgcolor(Color::Green) | color(Color::Black);

            row_elements.push_back(cell | size(WIDTH, EQUAL, 3) | center);
        }
        grid_rows.push_back(hbox(row_elements));
    }

    auto board_view = window(text(" Board "), vbox(grid_rows));

    auto key_up = text(" ↑/k ");
    if (state.last_input == "UP") key_up = key_up | inverted;
    auto key_down = text(" ↓/j ");
    if (state.last_input == "DOWN") key_down = key_down | inverted;
    auto key_left = text(" ←/h ");
    if (state.last_input == "LEFT") key_left = key_left | inverted;
    auto key_right = text(" →/l ");
    if (state.last_input == "RIGHT") key_right = key_right | inverted;

    auto controls_view = hbox({
        vbox({
            hbox({text("     "), key_up, text("     ")}),
            hbox({key_left, key_down, key_right})
        }),
        filler() | size(WIDTH, EQUAL, 4),
        vbox({
            text("[ENTER] Select / Move"),
            text("[ESC]   Deselect"),
            text("[C]     Toggle Casual"),
            text("[R]     Resign"),
            text("[Q]     Quit")
        })
    });

    auto status_view = vbox({
        text(state.status_msg) | color(Color::Green),
        state.game_over ? text("Game Over. Press 'q' to exit.") : text("Select a piece and move it to a highlighted square."),
        separator(),
        controls_view
    });

    return hbox({
        vbox({board_view, filler()}),
        text("  "), // small spacing
        vbox({window(text(" Info "), status_view) | size(WIDTH, EQUAL, 60), filler()})
    });
}

} // namespace stratego