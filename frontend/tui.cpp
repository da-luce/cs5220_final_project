#include "tui.h"
#include <ncurses.h>
#include <string>
#include <vector>

namespace tui {

void init_ncurses() {
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0); // Hide standard terminal cursor
    timeout(100); // 100ms timeout for getch() to allow key release detection

    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(1, COLOR_RED, -1);
        init_pair(2, COLOR_BLUE, -1);
        init_pair(3, COLOR_MAGENTA, -1);              // AI hidden
        init_pair(4, COLOR_CYAN, COLOR_BLUE);         // Water
        init_pair(5, COLOR_BLACK, COLOR_WHITE);       // Cursor Highlight
        init_pair(6, COLOR_BLACK, COLOR_YELLOW);      // Selected Piece
        init_pair(7, COLOR_GREEN, -1);                // Text Info
        init_pair(8, COLOR_BLACK, COLOR_GREEN);       // Valid Move (Empty)
        init_pair(9, COLOR_WHITE, COLOR_RED);         // Valid Move (Attack)
        init_pair(10, COLOR_GREEN, -1);               // Svelte 'v' checkmark
        init_pair(11, COLOR_CYAN, -1);                // Svelte '?' prompt
    }
}

bool show_start_menu(stratego::Board& board) {
    int stage = 0;
    
    int game_type_idx = 0;
    std::vector<std::string> game_types = {"Classic (10x10)", "Quick (8x8)", "Tiny (4x4)", "Load Game"};
    
    int layout_idx = 0;
    std::vector<std::string> layouts;
    
    int ai_idx = 0;
    std::vector<std::string> ais = {"Random AI"};
    
    stratego::GameType selected_game_type;
    stratego::SetupType selected_setup_type = stratego::SetupType::Default;

    while (true) {
        erase();
        mvprintw(2, 4, "=== STRATEGO SETUP ===");

        if (stage == 0) {
            attron(COLOR_PAIR(11) | A_BOLD);
            mvprintw(4, 4, "?");
            attroff(COLOR_PAIR(11) | A_BOLD);
            attron(A_BOLD);
            printw(" Select Game Type:");
            attroff(A_BOLD);

            for (size_t i = 0; i < game_types.size(); ++i) {
                if ((int)i == game_type_idx) {
                    attron(COLOR_PAIR(11));
                    mvprintw(6 + i, 4, "> ");
                    attroff(COLOR_PAIR(11));
                    attron(A_REVERSE);
                    printw("%s", game_types[i].c_str());
                    attroff(A_REVERSE);
                } else {
                    mvprintw(6 + i, 6, "%s", game_types[i].c_str());
                }
            }
            refresh();
            int ch = getch();
            if (ch == KEY_UP || ch == 'k') game_type_idx = (game_type_idx == 0) ? game_types.size() - 1 : game_type_idx - 1;
            else if (ch == KEY_DOWN || ch == 'j') game_type_idx = (game_type_idx + 1) % game_types.size();
            else if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) {
                if (game_type_idx == 3) {
                    char filename_c[256] = {0};
                    timeout(-1); curs_set(1); echo();
                    mvprintw(8 + game_types.size(), 4, "Enter filename: ");
                    refresh();
                    getstr(filename_c);
                    curs_set(0); noecho(); timeout(100);
                    if (board.load_from_file(filename_c) && std::string(filename_c).length() > 0) {
                        return true;
                    } else {
                        mvprintw(9 + game_types.size(), 4, "Error: Could not load file. Press any key.");
                        timeout(-1); getch(); timeout(100);
                    }
                } else {
                    if (game_type_idx == 0) {
                        selected_game_type = stratego::GameType::Classic;
                        layouts = {"Probabilistic (Dobby/Oewesok)", "Random"};
                    } else if (game_type_idx == 1) {
                        selected_game_type = stratego::GameType::Quick;
                        layouts = {"Random"};
                    } else {
                        selected_game_type = stratego::GameType::Tiny;
                        layouts = {"Random"};
                    }
                    layout_idx = 0;
                    stage = 1;
                }
            } else if (ch == 'q' || ch == 'Q') {
                return false;
            }
        } else if (stage == 1) {
            attron(COLOR_PAIR(10) | A_BOLD);
            mvprintw(4, 4, "v");
            attroff(COLOR_PAIR(10) | A_BOLD);
            printw(" Game Type: %s", game_types[game_type_idx].c_str());

            attron(COLOR_PAIR(11) | A_BOLD);
            mvprintw(6, 4, "?");
            attroff(COLOR_PAIR(11) | A_BOLD);
            attron(A_BOLD);
            printw(" Select Starting Layout:");
            attroff(A_BOLD);

            for (size_t i = 0; i < layouts.size(); ++i) {
                if ((int)i == layout_idx) {
                    attron(COLOR_PAIR(11));
                    mvprintw(8 + i, 4, "> ");
                    attroff(COLOR_PAIR(11));
                    attron(A_REVERSE);
                    printw("%s", layouts[i].c_str());
                    attroff(A_REVERSE);
                } else {
                    mvprintw(8 + i, 6, "%s", layouts[i].c_str());
                }
            }
            refresh();
            int ch = getch();
            if (ch == KEY_UP || ch == 'k') layout_idx = (layout_idx == 0) ? layouts.size() - 1 : layout_idx - 1;
            else if (ch == KEY_DOWN || ch == 'j') layout_idx = (layout_idx + 1) % layouts.size();
            else if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) {
                if (selected_game_type == stratego::GameType::Classic && layout_idx == 0) {
                    selected_setup_type = stratego::SetupType::Probabilistic;
                } else {
                    selected_setup_type = stratego::SetupType::Random;
                }
                stage = 2;
            } else if (ch == 27 || ch == KEY_BACKSPACE || ch == 127 || ch == 8 || ch == 'b' || ch == 'B') {
                stage = 0;
            } else if (ch == 'q' || ch == 'Q') {
                return false;
            }
        } else if (stage == 2) {
            attron(COLOR_PAIR(10) | A_BOLD);
            mvprintw(4, 4, "v");
            attroff(COLOR_PAIR(10) | A_BOLD);
            printw(" Game Type: %s", game_types[game_type_idx].c_str());

            attron(COLOR_PAIR(10) | A_BOLD);
            mvprintw(5, 4, "v");
            attroff(COLOR_PAIR(10) | A_BOLD);
            printw(" Starting Layout: %s", layouts[layout_idx].c_str());

            attron(COLOR_PAIR(11) | A_BOLD);
            mvprintw(7, 4, "?");
            attroff(COLOR_PAIR(11) | A_BOLD);
            attron(A_BOLD);
            printw(" Select AI Opponent:");
            attroff(A_BOLD);

            for (size_t i = 0; i < ais.size(); ++i) {
                if ((int)i == ai_idx) {
                    attron(COLOR_PAIR(11));
                    mvprintw(9 + i, 4, "> ");
                    attroff(COLOR_PAIR(11));
                    attron(A_REVERSE);
                    printw("%s", ais[i].c_str());
                    attroff(A_REVERSE);
                } else {
                    mvprintw(9 + i, 6, "%s", ais[i].c_str());
                }
            }
            refresh();
            int ch = getch();
            if (ch == KEY_UP || ch == 'k') ai_idx = (ai_idx == 0) ? ais.size() - 1 : ai_idx - 1;
            else if (ch == KEY_DOWN || ch == 'j') ai_idx = (ai_idx + 1) % ais.size();
            else if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) {
                board.initialize_game(selected_game_type, selected_setup_type);
                return true;
            } else if (ch == 27 || ch == KEY_BACKSPACE || ch == 127 || ch == 8 || ch == 'b' || ch == 'B') {
                stage = 1;
            } else if (ch == 'q' || ch == 'Q') {
                return false;
            }
        }
    }
    return true;
}

} // namespace tui