#include "stratego.h"
#include <ncurses.h>
#include <locale.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include <string>
#include "tui.h"
#include "stratego_ui.h"

using namespace stratego;

void append_combat_msg(CombatResult cr, std::string& msg, bool& game_over) {
    if (cr == CombatResult::AttackerWins) msg += " Attacker Wins!";
    else if (cr == CombatResult::DefenderWins) msg += " Defender Wins!";
    else if (cr == CombatResult::BothDestroyed) msg += " Both destroyed!";
    else if (cr == CombatResult::FlagCaptured) {
        msg += " FLAG CAPTURED!";
        game_over = true;
    }
}

void handle_confrontation(Board& board, GameState& state) {
    int ch = getch();
    if (ch == ERR) {
        state.last_ch = -1;
        return;
    }
    state.last_ch = ch;
    if (ch == 'q' || ch == 'Q') {
        state.exit_game = true;
    } else if (ch == '\n' || ch == '\r' || ch == ' ' || ch == KEY_ENTER) {
        state.in_confrontation = false;
        CombatResult res = board.execute_move(state.pending_move);
        state.status_msg = "Attack resolved.";
        append_combat_msg(res, state.status_msg, state.game_over);
        state.last_ch = -1;
    }
}

void handle_human_turn(Board& board, GameState& state) {
    int ch = getch();
    if (ch == ERR) { // No key was pressed in the last 100ms
        state.last_ch = -1;
        return;
    }
    
    state.last_ch = ch;
    int w = board.get_width();
    int h = board.get_height();

    if (state.last_ch == 'q' || state.last_ch == 'Q') state.exit_game = true;
    else if (state.last_ch == 'r' || state.last_ch == 'R') {
        state.game_over = true;
        state.status_msg = "You resigned. All enemy pieces revealed.";
        state.selected_x = -1;
        state.selected_y = -1;
        state.last_ch = -1;
    }
    else if (state.last_ch == 'c' || state.last_ch == 'C') {
        state.casual_mode = !state.casual_mode;
        state.status_msg = state.casual_mode ? "Casual mode ON (revealed pieces stay visible)." : "Casual mode OFF (perfect memory disabled).";
        state.last_ch = -1;
    }
    else if (state.last_ch == 's' || state.last_ch == 'S') {
        state.status_msg = "Save game as: ";
        move(15, 0); clrtoeol();
        mvprintw(15, 0, "%s", state.status_msg.c_str());

        char filename_c[256] = {0};
        timeout(-1); // Disable timeout for blocking input
        curs_set(1); // Show cursor for typing
        echo();
        mvgetstr(15, state.status_msg.length(), filename_c); // Get string at the prompt location
        noecho();
        curs_set(0); // Hide cursor again
        timeout(100); // Re-enable non-blocking getch

        if (std::string(filename_c).length() > 0 && board.save_to_file(filename_c)) {
            state.status_msg = "Game saved to " + std::string(filename_c) + ".";
        } else {
            state.status_msg = "Save cancelled or failed.";
        }
        state.last_ch = -1; // Prevent 'S' from staying highlighted
    }
    else if ((state.last_ch == KEY_UP || state.last_ch == 'k') && state.cursor_y > 0) state.cursor_y--;
    else if ((state.last_ch == KEY_DOWN || state.last_ch == 'j') && state.cursor_y < h - 1) state.cursor_y++;
    else if ((state.last_ch == KEY_LEFT || state.last_ch == 'h') && state.cursor_x > 0) state.cursor_x--;
    else if ((state.last_ch == KEY_RIGHT || state.last_ch == 'l') && state.cursor_x < w - 1) state.cursor_x++;
    else if (state.last_ch == '\n' || state.last_ch == '\r' || state.last_ch == ' ' || state.last_ch == KEY_ENTER) {
        if (state.selected_x == -1) {
            Piece p = board.get_piece(state.cursor_x, state.cursor_y);
            if (p.owner == Player::Red && p.is_mobile()) {
                state.selected_x = state.cursor_x;
                state.selected_y = state.cursor_y;
                state.status_msg = std::string(piece_name(p.type)) + " selected. Move to destination and press ENTER.";
            } else {
                state.status_msg = "Invalid piece! Select your own mobile piece.";
            }
        } else {
            if (state.selected_x == state.cursor_x && state.selected_y == state.cursor_y) {
                state.selected_x = -1; state.selected_y = -1;
                state.status_msg = "Piece deselected.";
            } else {
                Move m{state.selected_x, state.selected_y, state.cursor_x, state.cursor_y};
                if (board.is_legal_move(m)) {
                    Piece target = board.get_piece(state.cursor_x, state.cursor_y);
                    if (!target.is_empty()) {
                        state.in_confrontation = true;
                        state.pending_move = m;
                        state.pending_attacker_type = board.get_piece(state.selected_x, state.selected_y).type;
                        state.pending_defender_type = target.type;
                        state.status_msg = "Confrontation! Press ENTER to resolve.";
                        state.selected_x = -1; state.selected_y = -1;
                    } else {
                        CombatResult res = board.execute_move(m);
                        state.selected_x = -1; state.selected_y = -1;
                        state.status_msg = "You moved.";
                        append_combat_msg(res, state.status_msg, state.game_over);
                    }
                } else {
                    state.status_msg = "Illegal move! Try again or press ESC to deselect.";
                }
            }
        }
    } else if (state.last_ch == 27) { // ESC key
        state.selected_x = -1; state.selected_y = -1;
        state.status_msg = "Piece deselected.";
    }
}

void handle_ai_turn(Board& board, GameState& state, std::mt19937& gen) {
    state.status_msg = "AI is thinking...";
    attron(COLOR_PAIR(7));
    move(15, 0); clrtoeol();
    mvprintw(15, 0, "%s", state.status_msg.c_str());
    attroff(COLOR_PAIR(7));
    refresh();
    napms(600); // 600ms delay so AI move is visible
    
    std::vector<Move> legal_moves = board.get_all_legal_moves(Player::Blue);
    if (legal_moves.empty()) {
        state.status_msg = "VICTORY! The AI has no legal moves left.";
        state.game_over = true;
    } else {
        std::uniform_int_distribution<> dist(0, legal_moves.size() - 1);
        Move m = legal_moves[dist(gen)];
        
        Piece target = board.get_piece(m.end_x, m.end_y);
        if (!target.is_empty()) {
            state.in_confrontation = true;
            state.pending_move = m;
            state.pending_attacker_type = board.get_piece(m.start_x, m.start_y).type;
            state.pending_defender_type = target.type;
            state.status_msg = "AI Attacks! Press ENTER to resolve.";
        } else {
            CombatResult res = board.execute_move(m);
            state.status_msg = "AI moved (" + std::to_string(m.start_x) + "," + std::to_string(m.start_y) + ") -> (" + std::to_string(m.end_x) + "," + std::to_string(m.end_y) + ").";
            append_combat_msg(res, state.status_msg, state.game_over);
        }
    }
}

int main() {
    setlocale(LC_ALL, ""); // Ensure terminal supports full Unicode characters
    tui::init_ncurses();

    Board board;
    if (!show_start_menu(board)) {
        endwin();
        return 0;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    GameState state;
    state.cursor_x = 0;
    state.cursor_y = board.get_height() > 4 ? 6 : board.get_height() - 1;

    while (!state.exit_game) {
        render_board(board, state);

        if (state.game_over) {
            timeout(-1); // Restore blocking getch indefinitely for the exit screen
            int ch = getch();
            if (ch == 'q' || ch == 'Q') {
                break;
            }
            continue;
        }

        if (state.in_confrontation) {
            handle_confrontation(board, state);
            continue;
        }

        if (board.get_current_turn() == Player::Red) {
            handle_human_turn(board, state);
        } else {
            handle_ai_turn(board, state, gen);
        }
    }

    endwin(); // Restore standard terminal before quitting
    return 0;
}
