#include "policy_ucc.h"
#include "../stratego/engine.h"
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <string.h>
#include <vector>

namespace stratego {

// --- COORDINATE TRANSLATION HELPERS ---
// Red plays natively at the bottom. Blue thinks it plays at the bottom.
// We must mirror the Y-axis and UP/DOWN directions ONLY for Blue.

int PolicyUCC::to_bot_y(int abs_y) const {
    if (me_ == Player::Red) return abs_y;
    return local_board_.get_height() - 1 - abs_y;
}

int PolicyUCC::to_abs_y(int bot_y) const {
    if (me_ == Player::Red) return bot_y;
    return local_board_.get_height() - 1 - bot_y;
}

std::string PolicyUCC::to_bot_dir(const std::string& abs_dir) const {
    if (me_ == Player::Red) return abs_dir;
    if (abs_dir == "UP") return "DOWN";
    if (abs_dir == "DOWN") return "UP";
    return abs_dir;
}

std::string PolicyUCC::to_abs_dir(const std::string& bot_dir) const {
    if (me_ == Player::Red) return bot_dir;
    if (bot_dir == "UP") return "DOWN";
    if (bot_dir == "DOWN") return "UP";
    return bot_dir;
}
// --------------------------------------

PolicyUCC::PolicyUCC(const std::string& bot_path, Player me, GameState& state) 
    : me_(me), bot_path_(bot_path) {
    signal(SIGPIPE, SIG_IGN);
    local_board_ = state.board;
    start_process(state);
}

PolicyUCC::~PolicyUCC() {
    if (in_fd_ >= 0) {
        std::string color_str = (me_ == Player::Red) ? "RED" : "BLUE";
        std::cout << "MANAGER -> " << color_str << ": QUIT\n";
        std::string quit_msg = "QUIT\n";
        if(write(in_fd_, quit_msg.c_str(), quit_msg.length())) {}
    }
    
    if (pid_ > 0) {
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
    }
    if (in_fd_ >= 0) close(in_fd_);
    if (out_fd_ >= 0) close(out_fd_);
}

void PolicyUCC::send_line(const std::string& line) {
    std::string color_str = (me_ == Player::Red) ? "RED" : "BLUE";
    std::cout << "MANAGER -> " << color_str << ": " << line << "\n";
    
    std::string l = line + "\n";
    if (write(in_fd_, l.c_str(), l.length()) == -1) {
        throw std::runtime_error("Failed to write to bot pipe.");
    }
}

std::string PolicyUCC::read_line() {
    std::string result;
    char c;
    while (true) {
        ssize_t bytes_read = read(out_fd_, &c, 1);
        if (bytes_read <= 0) break;
        if (c == '\n') break;
        if (c != '\r') result += c;
    }
    
    std::string color_str = (me_ == Player::Red) ? "RED" : "BLUE";
    std::cout << color_str << " -> MANAGER: " << result << "\n";
    
    return result;
}

char PolicyUCC::piece_to_char(PieceType pt) const {
    switch (pt) {
        case PieceType::Empty: return '.';
        case PieceType::Water: return '+';
        case PieceType::Spy: return 's';
        case PieceType::Scout: return '9';
        case PieceType::Miner: return '8';
        case PieceType::Sergeant: return '7';
        case PieceType::Lieutenant: return '6';
        case PieceType::Captain: return '5';
        case PieceType::Major: return '4';
        case PieceType::Colonel: return '3';
        case PieceType::General: return '2';
        case PieceType::Marshal: return '1';
        case PieceType::Bomb: return 'B';
        case PieceType::Flag: return 'F';
        case PieceType::Hidden: return '#';
        default: return '?';
    }
}

PieceType PolicyUCC::char_to_piece(char c) const {
    switch (c) {
        case 's': return PieceType::Spy;
        case '9': return PieceType::Scout;
        case '8': return PieceType::Miner;
        case '7': return PieceType::Sergeant;
        case '6': return PieceType::Lieutenant;
        case '5': return PieceType::Captain;
        case '4': return PieceType::Major;
        case '3': return PieceType::Colonel;
        case '2': return PieceType::General;
        case '1': return PieceType::Marshal;
        case 'B': return PieceType::Bomb;
        case 'F': return PieceType::Flag;
        default: return PieceType::Empty;
    }
}

void PolicyUCC::send_board_snapshot() {
    const int w = local_board_.get_width();
    const int h = local_board_.get_height();
    
    // Iterate from the Bot's perspective (0 is Bot's Top/Enemy, 9 is Bot's Bottom/Self)
    for (int bot_y = 0; bot_y < h; ++bot_y) {
        int abs_y = to_abs_y(bot_y);
        std::string line;
        line.reserve(static_cast<size_t>(w));
        for (int x = 0; x < w; ++x) {
            Piece p = local_board_.get_piece(x, abs_y);
            if (p.type == PieceType::Empty) {
                line.push_back('.');
            } else if (p.type == PieceType::Water) {
                line.push_back('+');
            } else if (p.owner == me_) {
                line.push_back(piece_to_char(p.type));
            } else {
                line.push_back(p.revealed ? piece_to_char(p.type) : '#');
            }
        }
        send_line(line);
    }
}

std::string PolicyUCC::make_result_line(
    const Move& m,
    const Piece& attacker,
    const Piece& defender,
    const std::string& outcome
) const {
    std::string extra_info = "";

    // Send EXACTLY two tokens for ALL combat interactions to prevent parser hangs.
    if (!defender.is_empty() && outcome != "OK" && outcome != "NO_MOVE" && outcome != "FLAG") {
        extra_info = std::string(1, piece_to_char(attacker.type)) + " " + std::string(1, piece_to_char(defender.type));
    }

    std::string abs_dir = "UP";
    int mult = 1;
    if (m.end_x > m.start_x) { abs_dir = "RIGHT"; mult = m.end_x - m.start_x; }
    else if (m.end_x < m.start_x) { abs_dir = "LEFT"; mult = m.start_x - m.end_x; }
    else if (m.end_y > m.start_y) { abs_dir = "DOWN"; mult = m.end_y - m.start_y; }
    else if (m.end_y < m.start_y) { abs_dir = "UP"; mult = m.start_y - m.end_y; }

    int bot_start_y = to_bot_y(m.start_y);
    std::string bot_dir = to_bot_dir(abs_dir);

    std::string result = std::to_string(m.start_x) + " " + std::to_string(bot_start_y) + " " +
                         bot_dir + " " + std::to_string(mult) + " " + outcome;
                         
    if (!extra_info.empty()) {
        result += " " + extra_info;
    }
    
    return result;
}

void PolicyUCC::start_process(GameState& initial_state) {
    std::string actual_path = bot_path_;
    if (actual_path.front() != '/' && access(actual_path.c_str(), X_OK) != 0) {
        actual_path = "../" + actual_path;
    }

    if (access(actual_path.c_str(), X_OK) != 0) {
        throw std::runtime_error("Bot executable not found: " + actual_path);
    }

    int pipe_to_child[2], pipe_from_child[2];
    if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) throw std::runtime_error("Pipes failed");

    pid_ = fork();
    if (pid_ == 0) {
        dup2(pipe_to_child[0], STDIN_FILENO);
        dup2(pipe_from_child[1], STDOUT_FILENO);
        close(pipe_to_child[1]); close(pipe_from_child[0]);
        close(pipe_to_child[0]); close(pipe_from_child[1]);
        execl(actual_path.c_str(), actual_path.c_str(), nullptr);
        exit(1);
    } else {
        close(pipe_to_child[0]); close(pipe_from_child[1]);
        in_fd_ = pipe_to_child[1]; out_fd_ = pipe_from_child[0];

        if (me_ == Player::Red) std::cout << "\n=== SETUP PHASE ===\n";
        else std::cout << "\n";

        std::string bot_color = (me_ == Player::Red) ? "RED" : "BLUE";
        std::string opp_name = (me_ == Player::Red) ? "Blue_Destroyer" : "Red_Menace";
        
        std::string setup_msg = bot_color + " " + opp_name + " " + std::to_string(initial_state.board.get_width()) + " " + std::to_string(initial_state.board.get_height());
        send_line(setup_msg);

        const int h = initial_state.board.get_height();
        for (int y = 0; y < 4; y++) {
            std::string setup_line = read_line();
            // The bot expects these 4 lines to be its bottom rows (6, 7, 8, 9)
            int bot_y = (h - 4) + y; 
            int abs_y = to_abs_y(bot_y);
            
            for (size_t x = 0; x < setup_line.length() && x < initial_state.board.get_width(); x++) {
                PieceType type = char_to_piece(setup_line[x]);
                if (type != PieceType::Empty) {
                    local_board_.grid[local_board_.index(static_cast<int>(x), abs_y)] = Piece{type, me_, false};
                }
            }
        }
        initial_state.board = local_board_;

        if (me_ == Player::Red) {
            send_line("START");
        }
    }
}

Move PolicyUCC::get_move(const GameState& masked_state) {
    std::string color_str = (me_ == Player::Red) ? "RED" : "BLUE";
    std::cout << "\n=== TURN " << (masked_state.move_history.size() / 2) << ": " << color_str << " MOVES ===\n";

    // 1. Process moves sequentially to flawlessly reconstruct combat
    while (last_processed_move_ < (int)masked_state.move_history.size()) {
        Move m = masked_state.move_history[last_processed_move_];
        
        Piece attacker = local_board_.get_piece(m.start_x, m.start_y);
        Piece defender = local_board_.get_piece(m.end_x, m.end_y);
        Piece survivor = masked_state.board.get_piece(m.end_x, m.end_y);
        
        std::string outcome = "OK";
        if (!defender.is_empty()) {
            if (defender.type == PieceType::Flag) {
                outcome = "FLAG";
            } else if (survivor.is_empty() || survivor.type == PieceType::Empty) {
                outcome = "BOTHDIE";
            } else if (survivor.owner == attacker.owner) {
                outcome = "KILLS";
            } else {
                outcome = "DIES";
            }
        }

        // --- THE BEAUTIFUL FIX: DIRECT FROM ENGINE ---
        Piece res_atk = attacker;
        Piece res_def = defender;

        // If a fight happened, use the true identities recorded by the engine!
        if (outcome != "OK" && outcome != "NO_MOVE") {
            res_atk.type = m.attacker_type;
            res_def.type = m.defender_type;
            res_atk.revealed = true;
            res_def.revealed = true;
        }
        // ---------------------------------------------

        // Tell the bot what happened
        send_line(make_result_line(m, res_atk, res_def, outcome));

        // Update our sequential local board
        if (outcome == "KILLS" || outcome == "OK" || outcome == "FLAG") {
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = survivor.is_empty() ? res_atk : survivor;
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = Piece{PieceType::Empty, Player::None, false};
        } else if (outcome == "DIES") {
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = Piece{PieceType::Empty, Player::None, false};
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = survivor.is_empty() ? res_def : survivor;
        } else if (outcome == "BOTHDIE") {
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = Piece{PieceType::Empty, Player::None, false};
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = Piece{PieceType::Empty, Player::None, false};
        }
        
        last_processed_move_++;
    }

    // 2. NOW we can safely sync any newly revealed pieces from the masked state
    for (int y = 0; y < masked_state.board.get_height(); ++y) {
        for (int x = 0; x < masked_state.board.get_width(); ++x) {
            Piece p = masked_state.board.get_piece(x, y);
            if (p.type != PieceType::Hidden && p.type != PieceType::Empty) {
                local_board_.grid[local_board_.index(x, y)] = p;
            }
        }
    }

    send_board_snapshot();
    
    std::string move_line = read_line();
    std::stringstream ss(move_line);
    int bot_x, bot_y, mult = 1;
    std::string bot_dir;
    
    if (move_line == "NO_MOVE" || move_line == "SURRENDER") return Move{0, 0, 0, 0};

    if (!(ss >> bot_x >> bot_y >> bot_dir)) {
        throw std::runtime_error("Malformed move line: '" + move_line + "'");
    }
    if (!(ss >> mult)) mult = 1;

    int abs_y = to_abs_y(bot_y);
    std::string abs_dir = to_abs_dir(bot_dir);

    Move bot_move{bot_x, abs_y, bot_x, abs_y};
    if (abs_dir == "UP") bot_move.end_y -= mult;
    else if (abs_dir == "DOWN") bot_move.end_y += mult;
    else if (abs_dir == "LEFT") bot_move.end_x -= mult;
    else if (abs_dir == "RIGHT") bot_move.end_x += mult;

    return bot_move;
}

} // namespace stratego