#pragma once
#include "stratego.h"
#include <vector>

namespace stratego {

class RulesEngine {
public:
    static bool is_legal_move(const Board& board, const Move& move);
    static std::vector<Move> get_all_legal_moves(const Board& board, Player player);
    static std::vector<Move> get_legal_moves_for_piece(const Board& board, Player player, int x, int y);
};

} // namespace stratego