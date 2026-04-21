#include "setup_generator.h"
#include "stratego.h"
#include <fstream>
#include <stdexcept>
#include <random>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

// This implementation requires the nlohmann/json header-only library,
// which is managed by CMake's FetchContent.
#include "nlohmann/json.hpp"

namespace stratego::setup {

PieceDistribution load_distributions_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    nlohmann::json json_data;
    file >> json_data;

    PieceDistribution distributions;

    const std::map<std::string, PieceType> name_to_type = {
        {"SPY", PieceType::Spy}, {"SCOUT", PieceType::Scout},
        {"MINER", PieceType::Miner}, {"SERGEANT", PieceType::Sergeant},
        {"LIEUTENANT", PieceType::Lieutenant}, {"CAPTAIN", PieceType::Captain},
        {"MAJOR", PieceType::Major}, {"COLONEL", PieceType::Colonel},
        {"GENERAL", PieceType::General}, {"MARSHAL", PieceType::Marshal},
        {"BOMB", PieceType::Bomb}, {"FLAG", PieceType::Flag}
    };

    for (const auto& pair : name_to_type) {
        const std::string& name = pair.first;
        PieceType type = pair.second;

        if (json_data.contains(name)) {
            std::vector<std::vector<double>> grid(4, std::vector<double>(10));
            for (int i = 0; i < 4; ++i) {
                std::string row_key = std::to_string(i);
                if (json_data[name].contains(row_key)) {
                    grid[i] = json_data[name][row_key].get<std::vector<double>>();
                }
            }
            distributions[type] = grid;
        }
    }

    return distributions;
}

void generate_probabilistic_setup(Board& board, Player player, const PieceCounts& piece_counts, const PieceDistribution& distributions) {
    std::vector<PieceType> pieces_to_place;
    for (const auto& pair : piece_counts) {
        for (int i = 0; i < pair.second; ++i) {
            pieces_to_place.push_back(pair.first);
        }
    }

    // Shuffling ensures that we don't have a placement bias from iteration order
    // (e.g., always placing the Marshal first).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pieces_to_place.begin(), pieces_to_place.end(), gen);

    // Define the 4x10 deployment zone for the player.
    // This assumes a standard 10x10 board.
    std::vector<std::pair<int, int>> available_slots;
    int start_row = (player == Player::Blue) ? 0 : 6;
    for (int y = start_row; y < start_row + 4; ++y) {
        for (int x = 0; x < board.get_width(); ++x) {
            available_slots.emplace_back(x, y);
        }
    }

    // Iteratively place each piece
    for (const auto& piece_type : pieces_to_place) {
        if (available_slots.empty()) {
            throw std::runtime_error("Not enough available slots to place all pieces.");
        }

        std::vector<double> weights;
        weights.reserve(available_slots.size());

        // Build a list of weights for each available slot
        for (const auto& slot : available_slots) {
            int board_x = slot.first;
            int board_y = slot.second;

            // Map absolute board coordinates to player-relative 0-3 row index
            // for looking up probability in the distribution data.
            int relative_row;
            if (player == Player::Blue) {
                relative_row = board_y; // For Blue player, rows 0-3 map directly
            } else { // Player::Red
                relative_row = 9 - board_y; // For Red player, board row 9 is relative row 0, 8 is 1, etc.
            }

            double prob = 0.0;
            if (distributions.count(piece_type)) {
                // Add a small epsilon to avoid zero-probability issues if all remaining
                // slots have 0% chance, allowing it to still place the piece.
                prob = distributions.at(piece_type)[relative_row][board_x] + 1e-9;
            }
            weights.push_back(prob);
        }

        // Sample an index from the available slots based on the calculated weights
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        int chosen_slot_index = dist(gen);

        // Get the chosen slot's coordinates
        std::pair<int, int> chosen_slot = available_slots[chosen_slot_index];

        // Place the piece on the board
        board.place_piece(chosen_slot.first, chosen_slot.second, piece_type, player);

        // Remove the slot from the list of available slots
        available_slots.erase(available_slots.begin() + chosen_slot_index);
    }
}

} // namespace stratego::setup