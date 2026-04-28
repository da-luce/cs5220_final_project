#pragma once
#include "../stratego/policy.h"
#include "../stratego/board.h"
#include <string>
#include <vector>

namespace stratego {

class PolicyUCC : public Policy {
private:
    pid_t pid_ = -1;
    int in_fd_ = -1;
    int out_fd_ = -1;
    Player me_;
    std::string bot_path_;
    Board local_board_;
    int last_processed_move_ = 0;

    void start_process(GameState& initial_state);
    void send_board_snapshot();
    std::string make_result_line(const Move& m, const Piece& attacker, const Piece& defender, const std::string& outcome) const;
    void send_line(const std::string& line);
    std::string read_line();
    char piece_to_char(PieceType pt) const;
    PieceType char_to_piece(char c) const;

    // --- Coordinate Translation Helpers ---
    int to_bot_y(int abs_y) const;
    int to_abs_y(int bot_y) const;
    std::string to_bot_dir(const std::string& abs_dir) const;
    std::string to_abs_dir(const std::string& bot_dir) const;
    // --------------------------------------

public:
    PolicyUCC(const std::string& bot_path, Player me, GameState& state);
    ~PolicyUCC() override;

    Move get_move(const GameState& masked_state) override;
    bool is_human() const override { return false; }
};

} // namespace stratego