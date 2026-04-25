#include "../stratego/state.h"
#include "../stratego/engine.h"
#include "../runtime/game_runner.cpp"
#include "../runtime/player_human.cpp"
#include "../runtime/policy_random.cpp"
#include "../rl/policy_neural.h"
#include "../rl/networks/model.h"
#include "../rl/networks/torsos/cnn.h"
#include "../rl/encoding/board.h"

#include <iostream>
#include <string>
#include <optional>
#include <memory>
#include <thread>
#include <chrono>
#include "stratego_ui.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/event.hpp>

using namespace stratego;
using namespace ftxui;

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

std::optional<Move> process_human_input(Event event, const GameState& state, UIGameState& ui_state) {
    int w = state.board.get_width();
    int h = state.board.get_height();
    ui_state.last_input = "";

    if (ui_state.in_confrontation) {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            ui_state.exit_game = true;
        } else if (event == Event::Return || event == Event::Character(' ')) {
            ui_state.in_confrontation = false;
            return ui_state.pending_move;
        }
        return std::nullopt;
    }

    if (event == Event::Character('q') || event == Event::Character('Q')) ui_state.exit_game = true;
    else if (event == Event::Character('r') || event == Event::Character('R')) {
        ui_state.game_over = true;
        ui_state.status_msg = "You resigned. Game Over.";
        ui_state.selected_x = -1;
        ui_state.selected_y = -1;
    }
    else if (event == Event::Character('c') || event == Event::Character('C')) {
        ui_state.casual_mode = !ui_state.casual_mode;
        ui_state.status_msg = ui_state.casual_mode ? "Casual mode ON (revealed pieces stay visible)." : "Casual mode OFF.";
    }
    else if ((event == Event::ArrowUp || event == Event::Character('k')) && ui_state.cursor_y > 0) { ui_state.cursor_y--; ui_state.last_input = "UP"; }
    else if ((event == Event::ArrowDown || event == Event::Character('j')) && ui_state.cursor_y < h - 1) { ui_state.cursor_y++; ui_state.last_input = "DOWN"; }
    else if ((event == Event::ArrowLeft || event == Event::Character('h')) && ui_state.cursor_x > 0) { ui_state.cursor_x--; ui_state.last_input = "LEFT"; }
    else if ((event == Event::ArrowRight || event == Event::Character('l')) && ui_state.cursor_x < w - 1) { ui_state.cursor_x++; ui_state.last_input = "RIGHT"; }
    else if (event == Event::Return || event == Event::Character(' ')) {
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
    } else if (event == Event::Escape) {
        ui_state.selected_x = -1; ui_state.selected_y = -1;
        ui_state.status_msg = "Piece deselected.";
    }
    return std::nullopt;
}

int main() {
    auto settings_opt = start_menu();
    if (!settings_opt) {
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

    std::unique_ptr<Policy> red_agent = std::make_unique<Human>();
    std::unique_ptr<Policy> blue_agent;
    
    if (settings.ai_type == AIType::Random) {
        blue_agent = std::make_unique<Random>();
    } else {
        // Initialize NeuralPolicy
        int in_channels = get_encoding_channels(state.board.config);
        int hidden_filters = 64;

        encoding::actions::ActionEncoder encoder(state.board.config);
        int action_channels = encoder.get_action_channels();
        int H = state.board.config.height;
        int W = state.board.config.width;

        auto torso = std::make_shared<networks::torsos::CNNTorsoImpl>(in_channels, hidden_filters, 0);
        networks::StrategoNet model(torso, action_channels * H * W, H * W);

        try {
            torch::load(model, settings.model_path);
            blue_agent = std::make_unique<NeuralPolicy>(model, state.board.config);
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << "\n";
            return 1;
        }
    }

    GameRunner orch(state, std::move(red_agent), std::move(blue_agent));
    auto screen = ScreenInteractive::Fullscreen();
    bool ai_thinking = false;

    auto renderer = Renderer([&] {
        return render_board(orch.get_state(), ui_state);
    });

    auto catch_event = CatchEvent(renderer, [&](Event event) {
        if (ui_state.exit_game) {
            screen.Exit();
            return true;
        }
        
        if (ui_state.game_over) {
            if (event == Event::Character('q') || event == Event::Character('Q')) {
                screen.Exit();
                return true;
            }
            return false;
        }

        if (ai_thinking) return true; // Block input while AI is thinking

        Policy* active = orch.get_active_agent();
        if (active->is_human()) {
            auto m_opt = process_human_input(event, orch.get_state(), ui_state);
            if (m_opt) {
                static_cast<Human*>(active)->set_next_move(*m_opt);
                CombatResult res = orch.step();
                ui_state.status_msg = "Move resolved.";
                append_combat_msg(res, ui_state.status_msg, ui_state.game_over);
                
                // Immediately trigger AI turn if game is not over and it's AI turn
                if (!ui_state.game_over && !orch.get_active_agent()->is_human()) {
                    ai_thinking = true;
                    ui_state.status_msg = "AI is thinking...";
                    
                    auto task = [&screen]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(600));
                        screen.PostEvent(Event::Custom);
                    };
                    std::thread(task).detach();
                }
            }
            return true;
        }
        return false;
    });
    
    // We need to catch the Custom event which is fired by the AI thread
    auto final_component = CatchEvent(catch_event, [&](Event event) {
        if (event == Event::Custom && ai_thinking) {
            CombatResult res = orch.step();
            ui_state.status_msg = "AI moved.";
            append_combat_msg(res, ui_state.status_msg, ui_state.game_over);
            ai_thinking = false;
            return true;
        }
        return false;
    });

    // Check if AI goes first
    if (!orch.get_active_agent()->is_human()) {
        ai_thinking = true;
        ui_state.status_msg = "AI is thinking...";
        auto task = [&screen]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
            screen.PostEvent(Event::Custom);
        };
        std::thread(task).detach();
    }

    screen.Loop(final_component);

    return 0;
}