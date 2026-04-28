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
    // Translate from Engine Native (Red bottom) to UCC Native (Red top)
    Board ucc_board = local_board_.get_flipped_board();
    const int w = ucc_board.get_width();
    const int h = ucc_board.get_height();
    
    for (int y = 0; y < h; ++y) {
        std::string line;
        line.reserve(static_cast<size_t>(w));
        for (int x = 0; x < w; ++x) {
            Piece p = ucc_board.get_piece(x, y);
            if (p.type == PieceType::Empty) {
                line.push_back('.');
            } else if (p.type == PieceType::Water) {
                line.push_back('+');
            } else if (p.owner == me_) {
                line.push_back(piece_to_char(p.type));
            } else {
                // Per UCC spec, enemy pieces are always '#' — the bot tracks
                // revealed identities from result lines, not the snapshot.
                line.push_back('#');
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

    // Translate engine move to UCC global space
    // Engine is Red-bottom, UCC is Red-top, so we flip the board 180 degrees
    // get_flipped_move handles this coordinate transformation
    Move ucc_m = Engine::get_flipped_move(local_board_.get_config(), m); 

    std::string abs_dir = "UP";
    int mult = 1;
    if (ucc_m.end_x > ucc_m.start_x) { abs_dir = "RIGHT"; mult = ucc_m.end_x - ucc_m.start_x; }
    else if (ucc_m.end_x < ucc_m.start_x) { abs_dir = "LEFT"; mult = ucc_m.start_x - ucc_m.end_x; }
    else if (ucc_m.end_y > ucc_m.start_y) { abs_dir = "DOWN"; mult = ucc_m.end_y - ucc_m.start_y; }
    else if (ucc_m.end_y < ucc_m.start_y) { abs_dir = "UP"; mult = ucc_m.start_y - ucc_m.end_y; }

    std::string result = std::to_string(ucc_m.start_x) + " " + std::to_string(ucc_m.start_y) + " " +
                         abs_dir + " " + std::to_string(mult) + " " + outcome;
                         
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
        
        // 1. Get the current board translated into the UCC global space
        Board ucc_board = local_board_.get_flipped_board();

        for (int y = 0; y < 4; y++) {
            std::string setup_line = read_line();
            
            // 2. UCC Protocol dictates global placement:
            // RED is always at the top (rows 0, 1, 2, 3)
            // BLUE is always at the bottom (rows 6, 7, 8, 9)
            int ucc_y = (me_ == Player::Red) ? y : (h - 4) + y; 
            
            for (size_t x = 0; x < setup_line.length() && x < initial_state.board.get_width(); x++) {
                PieceType type = char_to_piece(setup_line[x]);
                if (type != PieceType::Empty) {
                    // 3. Place pieces directly onto the UCC board
                    ucc_board.grid[ucc_board.index(static_cast<int>(x), ucc_y)] = Piece{type, me_, false};
                }
            }
        }
        
        // 4. Translate the newly populated UCC board back to the engine's native space
        local_board_ = ucc_board.get_flipped_board();
        initial_state.board = local_board_;

        if (me_ == Player::Red) {
            send_line("START");
        }
    }
}

Move PolicyUCC::get_move(const GameState& masked_state) {
    std::string color_str = (me_ == Player::Red) ? "RED" : "BLUE";
    std::cout << "\n=== TURN " << (masked_state.move_history.size() / 2) << ": " << color_str << " MOVES ===\n";

    // 1. Process moves sequentially to flawlessly reconstruct combat.
    // Outcomes and survivors are derived from result_history — NOT from
    // masked_state.board, which reflects the state AFTER all queued moves
    // and would mis-report e.g. a DIES as BOTHDIE if the survivor was moved
    // away in a later turn.
    while (last_processed_move_ < (int)masked_state.move_history.size()) {
        Move m = masked_state.move_history[last_processed_move_];
        CombatResult r = masked_state.result_history[last_processed_move_];

        Piece attacker = local_board_.get_piece(m.start_x, m.start_y);
        Piece defender = local_board_.get_piece(m.end_x, m.end_y);

        std::string outcome;
        if (r.defender_type == PieceType::Empty) {
            outcome = "OK";
        } else if (r.defender_type == PieceType::Flag) {
            outcome = "FLAG";
        } else if (r.defender_type == PieceType::Bomb) {
            outcome = (r.attacker_type == PieceType::Miner) ? "KILLS" : "DIES";
        } else if (r.attacker_type == PieceType::Spy && r.defender_type == PieceType::Marshal) {
            outcome = "KILLS";
        } else {
            int a = static_cast<int>(r.attacker_type);
            int d = static_cast<int>(r.defender_type);
            outcome = (a > d) ? "KILLS" : (a < d) ? "DIES" : "BOTHDIE";
        }

        Piece res_atk = attacker;
        Piece res_def = defender;
        if (outcome != "OK") {
            res_atk.type = r.attacker_type;
            res_def.type = r.defender_type;
            res_atk.revealed = true;
            res_def.revealed = true;
        }

        send_line(make_result_line(m, res_atk, res_def, outcome));

        Piece empty_sq{PieceType::Empty, Player::None, false};
        if (outcome == "OK" || outcome == "KILLS" || outcome == "FLAG") {
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = res_atk;
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = empty_sq;
        } else if (outcome == "DIES") {
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = empty_sq;
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = res_def;
        } else if (outcome == "BOTHDIE") {
            local_board_.grid[local_board_.index(m.start_x, m.start_y)] = empty_sq;
            local_board_.grid[local_board_.index(m.end_x, m.end_y)] = empty_sq;
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
    int ucc_x, ucc_y, mult = 1;
    std::string ucc_dir;

    if (move_line == "NO_MOVE" || move_line == "SURRENDER") return Move{0, 0, 0, 0};

    if (!(ss >> ucc_x >> ucc_y >> ucc_dir)) {
        throw std::runtime_error("Malformed move line: '" + move_line + "'");
    }
    if (!(ss >> mult)) mult = 1;

    // Construct the move in UCC global space
    Move ucc_move{ucc_x, ucc_y, ucc_x, ucc_y};
    if (ucc_dir == "UP") ucc_move.end_y -= mult;
    else if (ucc_dir == "DOWN") ucc_move.end_y += mult;
    else if (ucc_dir == "LEFT") ucc_move.end_x -= mult;
    else if (ucc_dir == "RIGHT") ucc_move.end_x += mult;

    // Translate back to engine space
    return Engine::get_flipped_move(local_board_.get_config(), ucc_move);
}

} // namespace stratego