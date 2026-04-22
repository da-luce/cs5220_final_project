#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../controller/orchestrator.cpp"
#include "../agents/player_human.cpp"
#include "../agents/player_random.cpp"

#include <ncurses.h>
#include <locale.h>
#include <iostream>
#include <string>
#include <optional>
#include <memory>
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
    } else if (cr == CombatResult::Draw) {
        msg += " Game drawn.";
        game_over = true;
    }
}

std::optional<Move> process_human_input(const GameState& state, UIGameState& ui_state) {
    int ch = getch();
    if (ch == ERR) {
        ui_state.last_ch = -1;
        return std::nullopt;
    }
    
    ui_state.last_ch = ch;
    int w = state.board.get_width();
    int h = state.board.get_height();

    if (ui_state.in_confrontation) {
        if (ch == 'q' || ch == 'Q') {
            ui_state.exit_game = true;
        } else if (ch == '\n' || ch == '\r' || ch == ' ' || ch == KEY_ENTER) {
            ui_state.in_confrontation = false;
            ui_state.last_ch = -1;
            return ui_state.pending_move;
        }
        return std::nullopt;
    }

    if (ch == 'q' || ch == 'Q') ui_state.exit_game = true;
    else if (ch == 'r' || ch == 'R') {
        ui_state.game_over = true;
        ui_state.status_msg = "You resigned. Game Over.";
        ui_state.selected_x = -1;
        ui_state.selected_y = -1;
        ui_state.last_ch = -1;
    }
    else if (ch == 'c' || ch == 'C') {
        ui_state.casual_mode = !ui_state.casual_mode;
        ui_state.status_msg = ui_state.casual_mode ? "Casual mode ON (revealed pieces stay visible)." : "Casual mode OFF.";
        ui_state.last_ch = -1;
    }
    else if ((ch == KEY_UP || ch == 'k') && ui_state.cursor_y > 0) ui_state.cursor_y--;
    else if ((ch == KEY_DOWN || ch == 'j') && ui_state.cursor_y < h - 1) ui_state.cursor_y++;
    else if ((ch == KEY_LEFT || ch == 'h') && ui_state.cursor_x > 0) ui_state.cursor_x--;
    else if ((ch == KEY_RIGHT || ch == 'l') && ui_state.cursor_x < w - 1) ui_state.cursor_x++;
    else if (ch == '\n' || ch == '\r' || ch == ' ' || ch == KEY_ENTER) {
        if (ui_state.selected_x == -1) {
            Piece p = state.board.get_piece(ui_state.cursor_x, ui_state.cursor_y);
            if (p.owner == state.current_turn && p.is_mobile()) {
                ui_state.selected_x = ui_state.cursor_x;
                ui_state.selected_y = ui_state.cursor_y;
                ui_state.status_msg = std::string(piece_name(p.type)) + " selected. Move to destination and press ENTER.";
            } else {
                ui_state.status_msg = "Invalid piece! Select your own mobile piece.";
            }
        } else {
            if (ui_state.selected_x == ui_state.cursor_x && ui_state.selected_y == ui_state.cursor_y) {
                ui_state.selected_x = -1; ui_state.selected_y = -1;
                ui_state.status_msg = "Piece deselected.";
            } else {
                Move m{ui_state.selected_x, ui_state.selected_y, ui_state.cursor_x, ui_state.cursor_y};
                if (Engine::is_legal_move(state, m)) {
                    Piece target = state.board.get_piece(ui_state.cursor_x, ui_state.cursor_y);
                    if (!target.is_empty()) {
                        ui_state.in_confrontation = true;
                        ui_state.pending_move = m;
                        ui_state.pending_attacker_type = state.board.get_piece(ui_state.selected_x, ui_state.selected_y).type;
                        ui_state.pending_defender_type = target.type;
                        ui_state.status_msg = "Confrontation! Press ENTER to resolve.";
                        ui_state.selected_x = -1; ui_state.selected_y = -1;
                    } else {
                        ui_state.selected_x = -1; ui_state.selected_y = -1;
                        return m;
                    }
                } else {
                    ui_state.status_msg = "Illegal move! Try again or press ESC to deselect.";
                }
            }
        }
    } else if (ch == 27) { // ESC key
        ui_state.selected_x = -1; ui_state.selected_y = -1;
        ui_state.status_msg = "Piece deselected.";
    }
    return std::nullopt;
}

int main() {
    setlocale(LC_ALL, ""); // Ensure terminal supports full Unicode characters
    tui::init_ncurses();

    auto settings_opt = start_menu();
    if (!settings_opt) {
        endwin();
        return 0;
    }
    GameSettings settings = *settings_opt;

    GameState state;
    if (settings.new_game) {
        BoardConfig config = get_config_for_game_type(settings.game_type);
        state = state::initialize(config, settings.setup_type, 2000);
    } else {
        // Graceful fallback for non-supported save loadings right now
        BoardConfig config = get_config_for_game_type(GameType::Classic);
        state = state::initialize(config, state::SetupType::Random, 2000);
    }

    UIGameState ui_state;
    ui_state.cursor_x = 0;
    ui_state.cursor_y = state.board.get_height() > 4 ? 6 : state.board.get_height() - 1;
    ui_state.ai_type = (settings.ai_type == AIType::Random) ? "Random AI" : "Trained AI";
    ui_state.model_path = settings.model_path;

    std::unique_ptr<PlayerAgent> red_agent = std::make_unique<HumanPlayer>();
    std::unique_ptr<PlayerAgent> blue_agent = std::make_unique<RandomBot>(); 

    Orchestrator orch(state, std::move(red_agent), std::move(blue_agent));

    while (!ui_state.exit_game) {
        render_board(orch.get_state(), ui_state);

        if (ui_state.game_over) {
            timeout(-1); // Restore blocking getch indefinitely for the exit screen
            int ch = getch();
            if (ch == 'q' || ch == 'Q') {
                break;
            }
            continue;
        }

        PlayerAgent* active = orch.get_active_agent();

        if (active->is_human()) {
            auto m_opt = process_human_input(orch.get_state(), ui_state);
            if (m_opt) {
                static_cast<HumanPlayer*>(active)->set_next_move(*m_opt);
                CombatResult res = orch.step();
                ui_state.status_msg = "Move resolved.";
                append_combat_msg(res, ui_state.status_msg, ui_state.game_over);
            }
        } else {
            ui_state.status_msg = "AI is thinking...";
            render_board(orch.get_state(), ui_state);
            napms(600); // 600ms delay so AI move is visible

            CombatResult res = orch.step();
            ui_state.status_msg = "AI moved.";
            append_combat_msg(res, ui_state.status_msg, ui_state.game_over);
        }
    }

    endwin();
    return 0;
}
