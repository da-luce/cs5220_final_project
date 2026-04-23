#include "stratego_ui.h"
#include "tui.h"
#include <ncurses.h>
#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <filesystem>

namespace stratego {

const char* piece_to_str(PieceType type) {
    switch (type) {
        case PieceType::Spy: return "S ";
        case PieceType::Scout: return "2 ";
        case PieceType::Miner: return "3 ";
        case PieceType::Sergeant: return "4 ";
        case PieceType::Lieutenant: return "5 ";
        case PieceType::Captain: return "6 ";
        case PieceType::Major: return "7 ";
        case PieceType::Colonel: return "8 ";
        case PieceType::General: return "9 ";
        case PieceType::Marshal: return "★ "; // Star for Marshal
        case PieceType::Bomb: return "B ";
        case PieceType::Flag: return "⚑ ";    // Unicode Flag
        case PieceType::Water: return "≈≈";   // Water Waves
        default: return "· ";
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
    tui::Form form("=== STRATEGO SETUP ===");

    while (form.running()) {
        GameSettings settings;

        // 1. Select Game Type or Load
        std::string game_type = form.select("Select Game Type:", {
            "Classic (10x10)", "Barrage (10x10)", "Quick (8x8)", "Tiny (4x4)", "Load Game"
        });
        if (game_type.empty()) continue;

        // --- LOAD GAME PATH ---
        if (game_type == "Load Game") {
            std::string filename = form.text_input("Enter filename:");
            if (filename.empty()) continue;
            
            settings.new_game = false;
            settings.load_filename = filename;
            
            return settings;
        } 
        
        // --- NEW GAME PATH ---
        settings.new_game = true;
        
        settings.game_type = (game_type == "Classic (10x10)") ? GameType::Classic : 
                             (game_type == "Barrage (10x10)") ? GameType::Barrage :
                             (game_type == "Quick (8x8)")     ? GameType::Quick   : GameType::Tiny;

        // 2. Select Layout
        std::vector<std::string> layouts = {"Random"};
        if (settings.game_type == GameType::Classic) {
            layouts = {"Probabilistic (Dobby/Oewesok)", "Random"};
        }
        
        std::string layout = form.select("Select Starting Layout:", layouts);
        if (layout.empty()) continue;

        settings.setup_type = (layout == "Probabilistic (Dobby/Oewesok)") ? state::SetupType::Probabilistic : state::SetupType::Random;

        // 3. Select AI
        std::vector<std::string> ai_options = {"Random AI", "Trained AI"};

        std::string ai = form.select("Select AI Opponent:", ai_options);
        if (ai.empty()) continue;
        
        settings.ai_type = (ai == "Trained AI") ? AIType::Trained : AIType::Random;

        // 4. Model Path (If Trained AI)
        if (settings.ai_type == AIType::Trained) {
            std::string variant_str = (settings.game_type == GameType::Classic) ? "classic" :
                                      (settings.game_type == GameType::Barrage) ? "barrage" :
                                      (settings.game_type == GameType::Quick) ? "quick" : "tiny";
            std::string setup_str = (settings.setup_type == state::SetupType::Probabilistic) ? "probabilistic" :
                                    (settings.setup_type == state::SetupType::Default) ? "default" : "random";
            
            BoardConfig config = get_config_for_game_type(settings.game_type);
            std::string model_prefix = variant_str + "_" + setup_str;

            std::vector<std::string> model_files;
            if (std::filesystem::exists("models")) {
                for (const auto& entry : std::filesystem::directory_iterator("models")) {
                    if (entry.is_regular_file() && entry.path().extension() == ".pt") {
                        std::string filename = entry.path().filename().string();
                        if (filename == model_prefix + ".pt") {
                            model_files.push_back(entry.path().string());
                        }
                    }
                }
            }

            if (model_files.empty()) {
                form.restart_with_error("No models found matching: " + model_prefix + ".pt\nTrain one with: ./build/rl/train_stratego " + variant_str + " " + setup_str + " <num_episodes>");
                continue; // Restart loop
            }

            std::string selected_model = form.select("Select Model:", model_files);
            if (selected_model.empty()) continue;
            
            settings.model_path = selected_model;
        }

        return settings;
    }
    
    // Returns empty if the user quit the menu (e.g., pressed ESC)
    return std::nullopt; 
}

void render_board(const GameState& game_state, const UIGameState& state) {
    erase(); // Use erase() instead of clear() to avoid flickering during 100ms redraws
    mvprintw(0, 0, "=== STRATEGO ===");
    
    const Board& board = game_state.board;
    std::vector<Move> active_legal_moves;
    if (state.selected_x != -1 && state.selected_y != -1) {
        active_legal_moves = Engine::get_legal_moves_for_piece(game_state, state.selected_x, state.selected_y);
    }
    
    int w = board.get_width();
    int h = board.get_height();

    mvprintw(1, 3, "┌");
    for (int x = 0; x < w; ++x) printw("───");
    printw("┐");

    for (int y = 0; y < h; ++y) {
        mvprintw(2 + y, 0, " %d │", y);
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

            int color_pair = 0;
            bool is_confrontation_target = (state.in_confrontation && x == state.pending_move.end_x && y == state.pending_move.end_y);
            bool is_confrontation_source = (state.in_confrontation && x == state.pending_move.start_x && y == state.pending_move.start_y);

            if (is_selected) color_pair = 6;
            else if (is_cursor) color_pair = 5;
            else if (is_valid_attack) color_pair = 9;
            else if (is_valid_empty) color_pair = 8;
            else if (is_confrontation_source) color_pair = 0;
            else if (is_confrontation_target) color_pair = 0;
            else if (p.is_obstacle()) color_pair = 4;
            else if (!p.is_empty()) {
                if (p.owner == Player::Red) color_pair = 1;
                else color_pair = (state.casual_mode && p.revealed) ? 2 : 3;
                if (state.game_over && p.owner == Player::Blue) color_pair = 2;
            }

            if (color_pair) attron(COLOR_PAIR(color_pair));

            if (is_confrontation_source) {
                printw("·  ");
            } else if (is_confrontation_target) {
                std::string a_str = piece_to_str(state.pending_attacker_type);
                if (!a_str.empty() && a_str.back() == ' ') a_str.pop_back();
                std::string d_str = piece_to_str(state.pending_defender_type);
                if (!d_str.empty() && d_str.back() == ' ') d_str.pop_back();

                Player current_player = game_state.current_turn;
                int attacker_color = (current_player == Player::Red) ? 1 : 2;
                int defender_color = (current_player == Player::Red) ? 2 : 1;

                if (color_pair) attroff(COLOR_PAIR(color_pair));
                attron(COLOR_PAIR(attacker_color));
                printw("%s", a_str.c_str());
                attroff(COLOR_PAIR(attacker_color));
                attron(COLOR_PAIR(defender_color));
                printw("%s", d_str.c_str());
                attroff(COLOR_PAIR(defender_color));
                if (color_pair) attron(COLOR_PAIR(color_pair));
                printw(" ");
            }
            else if (p.is_obstacle()) printw("≈≈ ");
            else if (p.is_empty()) printw("·  ");
            else if (p.owner == Player::Red) printw("%s ", piece_to_str(p.type));
            else {
                if (state.game_over || (state.casual_mode && p.revealed)) printw("%s ", piece_to_str(p.type));
                else printw("?  ");
            }

            if (color_pair) attroff(COLOR_PAIR(color_pair));
        }
        printw("│");
    }
    mvprintw(2 + h, 3, "└");
    for (int x = 0; x < w; ++x) printw("───");
    printw("┘");

    mvprintw(3 + h, 4, "");
    for (int x = 0; x < w; ++x) {
        if (x == 0) printw(" %d", x);
        else printw("  %d", x);
    }

    attron(COLOR_PAIR(7));
    move(15, 0); clrtoeol();
    mvprintw(15, 0, "%s", state.status_msg.c_str());
    move(16, 0); clrtoeol();
    if (state.game_over) mvprintw(16, 0, "Game Over. Press 'q' to exit.");
    else mvprintw(16, 0, "Select a piece and move it to a highlighted square.");
    attroff(COLOR_PAIR(7));

    mvprintw(18, 4, "       "); if (state.last_ch == KEY_UP || state.last_ch == 'k') attron(A_REVERSE); printw("↑/k"); if (state.last_ch == KEY_UP || state.last_ch == 'k') attroff(A_REVERSE);
    move(19, 4); printw(" "); if (state.last_ch == KEY_LEFT || state.last_ch == 'h') attron(A_REVERSE); printw("←/h"); if (state.last_ch == KEY_LEFT || state.last_ch == 'h') attroff(A_REVERSE); printw("   ");
    if (state.last_ch == KEY_DOWN || state.last_ch == 'j') attron(A_REVERSE); printw("↓/j"); if (state.last_ch == KEY_DOWN || state.last_ch == 'j') attroff(A_REVERSE); printw("   ");
    if (state.last_ch == KEY_RIGHT || state.last_ch == 'l') attron(A_REVERSE); printw("→/l"); if (state.last_ch == KEY_RIGHT || state.last_ch == 'l') attroff(A_REVERSE);
    mvprintw(18, 25, "[ENTER] Select / Move"); mvprintw(19, 25, "[ESC]   Deselect"); mvprintw(20, 25, "[S]     Save Game"); mvprintw(21, 25, "[C]     Toggle Casual"); mvprintw(22, 25, "[R]     Resign"); mvprintw(23, 25, "[Q]     Quit"); refresh();
}

} // namespace stratego