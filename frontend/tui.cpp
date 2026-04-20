#include "tui.h"
#include <ncurses.h>
#include <string>

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
    }
}

bool show_start_menu(stratego::Board& board) {
    bool game_started = false;
    int menu_highlight = 0;
    const char *choices[] = {"Start Normal Game", "Start Tiny Game", "Start Quick Game", "Load Game", "Quit"};
    int n_choices = sizeof(choices) / sizeof(char *);

    while (!game_started) {
        erase();
        mvprintw(4, 10, "=== STRATEGO NCURSES ===");
        for (int i = 0; i < n_choices; ++i) {
            if (menu_highlight == i) {
                attron(A_REVERSE);
                mvprintw(6 + i, 12, "> %s <", choices[i]);
                attroff(A_REVERSE);
            } else {
                mvprintw(6 + i, 14, "%s", choices[i]);
            }
        }
        refresh();

        int ch = getch();
        switch(ch) {
            case KEY_UP:    menu_highlight = (menu_highlight == 0) ? n_choices - 1 : menu_highlight - 1; break;
            case KEY_DOWN:  menu_highlight = (menu_highlight + 1) % n_choices; break;
            case '\n': case '\r': case KEY_ENTER:
                if (menu_highlight == 0) { // New Normal Game
                    board.initialize_game(stratego::GameType::Normal);
                    game_started = true;
                } else if (menu_highlight == 1) { // New Tiny Game
                    board.initialize_game(stratego::GameType::Tiny);
                    game_started = true;
                } else if (menu_highlight == 2) { // New Quick Game
                    board.initialize_game(stratego::GameType::Quick);
                    game_started = true;
                } else if (menu_highlight == 3) { // Load Game
                    char filename_c[256] = {0};
                    timeout(-1); // Disable timeout to allow blocking input
                    curs_set(1); // Show cursor for typing
                    echo(); // Show user input
                    mvprintw(12, 10, "Enter filename: ");
                    refresh(); // Ensure prompt is displayed before waiting for input
                    getstr(filename_c);
                    curs_set(0); // Hide cursor again
                    noecho(); // Hide user input again
                    timeout(100); // Re-enable timeout for main game loop
                    if (board.load_from_file(filename_c) && std::string(filename_c).length() > 0) {
                        game_started = true;
                    } else {
                        mvprintw(13, 10, "Error: Could not load file. Press any key.");
                        timeout(-1); getch(); timeout(100); // Wait for keypress before continuing
                    }
                } else if (menu_highlight == 4) { 
                    return false; // Quit
                }
                break;
        }
    }
    return true;
}

} // namespace tui