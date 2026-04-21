#include "game_serializer.h"
#include <fstream>
#include <vector>
#include <string>

namespace stratego {

bool GameSerializer::save_to_file(const Board& board, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;

    // 1. Write metadata (dimensions, current turn)
    ofs.write(reinterpret_cast<const char*>(&board.config.width), sizeof(board.config.width));
    ofs.write(reinterpret_cast<const char*>(&board.config.height), sizeof(board.config.height));
    char turn = static_cast<char>(board.current_turn);
    ofs.write(&turn, sizeof(turn));

    // 2. Write grid data piece by piece
    for (const auto& piece : board.grid) {
        char type = static_cast<char>(piece.type);
        char owner = static_cast<char>(piece.owner);
        char revealed = piece.revealed ? 1 : 0;
        ofs.write(&type, sizeof(type));
        ofs.write(&owner, sizeof(owner));
        ofs.write(&revealed, sizeof(revealed));
    }

    // 3. Write move count and move history
    ofs.write(reinterpret_cast<const char*>(&board.move_count), sizeof(board.move_count));
    size_t history_size = board.move_history.size();
    ofs.write(reinterpret_cast<const char*>(&history_size), sizeof(history_size));
    if (history_size > 0) {
        ofs.write(reinterpret_cast<const char*>(board.move_history.data()), history_size * sizeof(Move));
    }

    // 4. Write chase hashes
    auto write_hashes = [&ofs](const std::vector<std::string>& hashes) {
        size_t size = hashes.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& hash : hashes) {
            size_t len = hash.length();
            ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
            ofs.write(hash.data(), len);
        }
    };
    write_hashes(board.red_chase_hashes);
    write_hashes(board.blue_chase_hashes);

    return ofs.good();
}

bool GameSerializer::load_from_file(Board& board, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) return false;

    // 1. Read metadata
    int new_width, new_height;
    ifs.read(reinterpret_cast<char*>(&new_width), sizeof(new_width));
    ifs.read(reinterpret_cast<char*>(&new_height), sizeof(new_height));
    char turn;
    ifs.read(&turn, sizeof(turn));
    if (ifs.fail()) return false;

    // 2. Update board state
    board.config.width = new_width;
    board.config.height = new_height;
    board.current_turn = static_cast<Player>(turn);
    board.move_count = 0;
    board.move_history.clear();
    board.red_chase_hashes.clear();
    board.blue_chase_hashes.clear();
    board.grid.resize(board.config.width * board.config.height);

    // 3. Read grid data
    for (auto& piece : board.grid) {
        char type_c, owner_c, revealed_c;
        ifs.read(&type_c, sizeof(type_c));
        ifs.read(&owner_c, sizeof(owner_c));
        ifs.read(&revealed_c, sizeof(revealed_c));
        if (ifs.fail()) return false;
        piece = {static_cast<PieceType>(type_c), static_cast<Player>(owner_c), (revealed_c != 0)};
    }

    // 4. Read extended state if available
    if (ifs.peek() != EOF) {
        ifs.read(reinterpret_cast<char*>(&board.move_count), sizeof(board.move_count));
        size_t history_size = 0;
        ifs.read(reinterpret_cast<char*>(&history_size), sizeof(history_size));
        
        if (!ifs.fail()) {
            board.move_history.resize(history_size);
            if (history_size > 0) {
                ifs.read(reinterpret_cast<char*>(board.move_history.data()), history_size * sizeof(Move));
            }

            // 5. Read chase hashes
            auto read_hashes = [&ifs](std::vector<std::string>& hashes) {
                size_t size = 0;
                ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (ifs.fail()) return;
                hashes.resize(size);
                for (size_t i = 0; i < size; ++i) {
                    size_t len = 0;
                    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
                    if (ifs.fail()) return;
                    std::string hash(len, '\0');
                    ifs.read(&hash[0], len);
                    hashes[i] = hash;
                }
            };
            read_hashes(board.red_chase_hashes);
            read_hashes(board.blue_chase_hashes);
        }
    }
    
    return true; // We can assume success if we reach here
}

} // namespace stratego