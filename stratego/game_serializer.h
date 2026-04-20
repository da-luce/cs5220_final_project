#pragma once
#include "stratego.h"
#include <string>

namespace stratego {

class GameSerializer {
public:
    static bool save_to_file(const Board& board, const std::string& filename);
    static bool load_from_file(Board& board, const std::string& filename);
};

} // namespace stratego