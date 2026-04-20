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

int prompt_list(const std::string& title, 
                const std::vector<std::pair<std::string, std::string>>& history, 
                const std::string& question, 
                const std::vector<std::string>& choices) {
    int highlight = 0;
    while (true) {
        erase();
        int row = 2;
        if (!title.empty()) {
            mvprintw(row++, 4, "%s", title.c_str());
            row++;
        }

        for (const auto& past : history) {
            attron(COLOR_PAIR(10) | A_BOLD);
            mvprintw(row, 4, "v");
            attroff(COLOR_PAIR(10) | A_BOLD);
            printw(" %s %s", past.first.c_str(), past.second.c_str());
            row++;
        }
        if (!history.empty()) row++;

        attron(COLOR_PAIR(11) | A_BOLD);
        mvprintw(row, 4, "?");
        attroff(COLOR_PAIR(11) | A_BOLD);
        attron(A_BOLD);
        printw(" %s", question.c_str());
        attroff(A_BOLD);
        row += 2;

        for (size_t i = 0; i < choices.size(); ++i) {
            if ((int)i == highlight) {
                attron(COLOR_PAIR(11));
                mvprintw(row + i, 4, "> ");
                attroff(COLOR_PAIR(11));
                attron(A_REVERSE);
                printw("%s", choices[i].c_str());
                attroff(A_REVERSE);
            } else {
                mvprintw(row + i, 6, "%s", choices[i].c_str());
            }
        }
        refresh();

        int ch = getch();
        if (ch == KEY_UP || ch == 'k') highlight = (highlight == 0) ? choices.size() - 1 : highlight - 1;
        else if (ch == KEY_DOWN || ch == 'j') highlight = (highlight + 1) % choices.size();
        else if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) return highlight;
        else if (ch == 27 || ch == KEY_BACKSPACE || ch == 127 || ch == 8 || ch == 'b' || ch == 'B') return -1;
        else if (ch == 'q' || ch == 'Q') return -2;
    }
}

int prompt_input(const std::string& title,
                 const std::vector<std::pair<std::string, std::string>>& history,
                 const std::string& question,
                 std::string& out_str) {
    erase();
    int row = 2;
    if (!title.empty()) {
        mvprintw(row++, 4, "%s", title.c_str());
        row++;
    }

    for (const auto& past : history) {
        attron(COLOR_PAIR(10) | A_BOLD);
        mvprintw(row, 4, "v");
        attroff(COLOR_PAIR(10) | A_BOLD);
        printw(" %s %s", past.first.c_str(), past.second.c_str());
        row++;
    }
    if (!history.empty()) row++;

    attron(COLOR_PAIR(11) | A_BOLD);
    mvprintw(row, 4, "?");
    attroff(COLOR_PAIR(11) | A_BOLD);
    attron(A_BOLD);
    printw(" %s ", question.c_str());
    attroff(A_BOLD);

    refresh();

    char buffer[256] = {0};
    timeout(-1); 
    curs_set(1); 
    echo();
    
    getstr(buffer);
    
    curs_set(0); 
    noecho(); 
    timeout(100);

    out_str = buffer;
    if (out_str.empty()) return -1;
    return 1;
}

void show_error(const std::string& msg) {
    erase();
    attron(COLOR_PAIR(1) | A_BOLD);
    mvprintw(10, 4, "%s", msg.c_str());
    attroff(COLOR_PAIR(1) | A_BOLD);
    mvprintw(12, 4, "Press any key to try again.");
    refresh();
    timeout(-1); getch(); timeout(100);
}

} // namespace tui