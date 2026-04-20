#include "stratego.h"
#include <ncurses.h>
#include <locale.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

using namespace stratego;

// Map pieces to Unicode-friendly 2-character wide strings
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

void setup_random_army(Board& board, Player player, int start_y) {
    std::vector<PieceType> deck = {
        PieceType::Flag, PieceType::Marshal, PieceType::General, PieceType::Spy
    };
    deck.insert(deck.end(), 6, PieceType::Bomb);
    deck.insert(deck.end(), 8, PieceType::Scout);
    deck.insert(deck.end(), 5, PieceType::Miner);
    deck.insert(deck.end(), 4, PieceType::Sergeant);
    deck.insert(deck.end(), 4, PieceType::Lieutenant);
    deck.insert(deck.end(), 4, PieceType::Captain);
    deck.insert(deck.end(), 3, PieceType::Major);
    deck.insert(deck.end(), 2, PieceType::Colonel);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(deck.begin(), deck.end(), g);

    int idx = 0;
    for (int y = start_y; y < start_y + 4; ++y) {
        for (int x = 0; x < 10; ++x) {
            board.place_piece(x, y, deck[idx++], player);
        }
    }
}

int main() {
    setlocale(LC_ALL, ""); // Ensure terminal supports full Unicode characters
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

    Board board;
    setup_random_army(board, Player::Blue, 0);
    setup_random_army(board, Player::Red, 6);

    std::random_device rd;
    std::mt19937 gen(rd());

    int cursor_x = 0, cursor_y = 6;
    int selected_x = -1, selected_y = -1;
    bool exit_game = false;
    bool game_over = false;
    int last_ch = -1;
    std::string status_msg = "Welcome! You are RED. Use Arrows to move, ENTER to select.";

    auto append_combat_msg = [&](CombatResult cr, std::string& msg) {
        if (cr == CombatResult::AttackerWins) msg += " Attacker Wins!";
        else if (cr == CombatResult::DefenderWins) msg += " Defender Wins!";
        else if (cr == CombatResult::BothDestroyed) msg += " Both destroyed!";
        else if (cr == CombatResult::FlagCaptured) {
            msg += " FLAG CAPTURED!";
            game_over = true;
        }
    };

    while (!exit_game) {
        erase(); // Use erase() instead of clear() to avoid flickering during 100ms redraws
        mvprintw(0, 0, "=== STRATEGO ===");
        
        std::vector<Move> active_legal_moves;
        if (selected_x != -1 && selected_y != -1) {
            active_legal_moves = board.get_legal_moves_for_piece(Player::Red, selected_x, selected_y);
        }
        
        // Render the ASCII Grid
        mvprintw(1, 3, "┌──────────────────────────────┐");
        for (int y = 0; y < 10; ++y) {
            mvprintw(2 + y, 0, " %d │", y);
            for (int x = 0; x < 10; ++x) {
                Piece p = board.get_piece(x, y);
                bool is_cursor = (x == cursor_x && y == cursor_y) && !game_over;
                bool is_selected = (x == selected_x && y == selected_y);

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
                if (is_selected) color_pair = 6;
                else if (is_cursor) color_pair = 5;
                else if (is_valid_attack) color_pair = 9;
                else if (is_valid_empty) color_pair = 8;
                else if (p.is_obstacle()) color_pair = 4;
                else if (!p.is_empty()) {
                    if (p.owner == Player::Red) color_pair = 1;
                    else color_pair = p.revealed ? 2 : 3;
                }

                if (color_pair) attron(COLOR_PAIR(color_pair));

                if (p.is_obstacle()) printw("≈≈ ");
                else if (p.is_empty()) printw("·  ");
                else if (p.owner == Player::Red) printw("%s ", piece_to_str(p.type));
                else {
                    if (p.revealed || game_over) printw("%s ", piece_to_str(p.type));
                    else printw("?  ");
                }

                if (color_pair) attroff(COLOR_PAIR(color_pair));
            }
            printw("│");
        }
        mvprintw(12, 3, "└──────────────────────────────┘");
        mvprintw(13, 4, " 0  1  2  3  4  5  6  7  8  9");

        // Print dynamic messages below the board
        attron(COLOR_PAIR(7));
        move(15, 0); clrtoeol();
        mvprintw(15, 0, "%s", status_msg.c_str());
        move(16, 0); clrtoeol();
        if (game_over) {
            mvprintw(16, 0, "Game Over. Press any key to exit.");
        } else {
            mvprintw(16, 0, "Select a piece and move it to a highlighted square.");
        }
        attroff(COLOR_PAIR(7));

        mvprintw(18, 4, "       ");
        if (last_ch == KEY_UP || last_ch == 'k') attron(A_REVERSE);
        printw("↑/k");
        if (last_ch == KEY_UP || last_ch == 'k') attroff(A_REVERSE);

        move(19, 4);
        printw(" ");
        if (last_ch == KEY_LEFT || last_ch == 'h') attron(A_REVERSE);
        printw("←/h");
        if (last_ch == KEY_LEFT || last_ch == 'h') attroff(A_REVERSE);
        printw("   ");

        if (last_ch == KEY_DOWN || last_ch == 'j') attron(A_REVERSE);
        printw("↓/j");
        if (last_ch == KEY_DOWN || last_ch == 'j') attroff(A_REVERSE);
        printw("   ");

        if (last_ch == KEY_RIGHT || last_ch == 'l') attron(A_REVERSE);
        printw("→/l");
        if (last_ch == KEY_RIGHT || last_ch == 'l') attroff(A_REVERSE);

        mvprintw(18, 25, "[ENTER] Select / Move");
        mvprintw(19, 25, "[ESC]   Deselect");
        mvprintw(20, 25, "[Q]     Quit");
        refresh();

        if (game_over) {
            timeout(-1); // Restore blocking getch indefinitely for the exit screen
            getch();
            break;
        }

        CombatResult res = CombatResult::MovedToEmpty;
        
        if (board.get_current_turn() == Player::Red) {
            int ch = getch();
            if (ch == ERR) { // No key was pressed in the last 100ms
                last_ch = -1;
                continue;
            }
            
            last_ch = ch;
            if (last_ch == 'q' || last_ch == 'Q') exit_game = true;
            else if ((last_ch == KEY_UP || last_ch == 'k') && cursor_y > 0) cursor_y--;
            else if ((last_ch == KEY_DOWN || last_ch == 'j') && cursor_y < 9) cursor_y++;
            else if ((last_ch == KEY_LEFT || last_ch == 'h') && cursor_x > 0) cursor_x--;
            else if ((last_ch == KEY_RIGHT || last_ch == 'l') && cursor_x < 9) cursor_x++;
            else if (last_ch == '\n' || last_ch == '\r' || last_ch == ' ' || last_ch == KEY_ENTER) {
                if (selected_x == -1) {
                    Piece p = board.get_piece(cursor_x, cursor_y);
                    if (p.owner == Player::Red && p.is_mobile()) {
                        selected_x = cursor_x;
                        selected_y = cursor_y;
                        status_msg = "Piece selected. Move to destination and press ENTER.";
                    } else {
                        status_msg = "Invalid piece! Select your own mobile piece.";
                    }
                } else {
                    if (selected_x == cursor_x && selected_y == cursor_y) {
                        selected_x = -1; selected_y = -1;
                        status_msg = "Piece deselected.";
                    } else {
                        Move m{selected_x, selected_y, cursor_x, cursor_y};
                        if (board.is_legal_move(m)) {
                            res = board.execute_move(m);
                            selected_x = -1; selected_y = -1;
                            status_msg = "You moved.";
                            append_combat_msg(res, status_msg);
                        } else {
                            status_msg = "Illegal move! Try again or press ESC to deselect.";
                        }
                    }
                }
            } else if (last_ch == 27) { // ESC key
                selected_x = -1; selected_y = -1;
                status_msg = "Piece deselected.";
            }
        } else {
            status_msg = "AI is thinking...";
            attron(COLOR_PAIR(7));
            move(15, 0); clrtoeol();
            mvprintw(15, 0, "%s", status_msg.c_str());
            attroff(COLOR_PAIR(7));
            refresh();
            napms(600); // 600ms delay so AI move is visible
            
            std::vector<Move> legal_moves = board.get_all_legal_moves(Player::Blue);
            if (legal_moves.empty()) {
                status_msg = "VICTORY! The AI has no legal moves left.";
                game_over = true;
            } else {
                std::uniform_int_distribution<> dist(0, legal_moves.size() - 1);
                Move m = legal_moves[dist(gen)];
                res = board.execute_move(m);
                
                status_msg = "AI moved (" + std::to_string(m.start_x) + "," + std::to_string(m.start_y) + ") -> (" + std::to_string(m.end_x) + "," + std::to_string(m.end_y) + ").";
                append_combat_msg(res, status_msg);
            }
        }
    }

    endwin(); // Restore standard terminal before quitting
    return 0;
}
