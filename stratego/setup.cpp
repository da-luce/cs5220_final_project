#include "setup.h"
#include <fstream>
#include <stdexcept>
#include <random>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

// This implementation requires the nlohmann/json header-only library
#include "nlohmann/json.hpp"

namespace stratego::setup {

void normalize_grid(std::vector<std::vector<double>>& grid) {
    double sum = 0.0;
    for (const auto& row : grid) {
        for (double p : row) {
            sum += p;
        }
    }
    
    // Normalize to sum exactly to 1.0, avoiding division by zero
    if (sum > 0.0) {
        for (auto& row : grid) {
            for (double& p : row) {
                p /= sum;
            }
        }
    }
}

PieceDistribution load_4x10_distributions_from_json(const std::string& filename) {
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
            std::vector<std::vector<double>> grid(4, std::vector<double>(10, 0.0));
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

// Do bilinear interpolation to map the 4x10 distributions to the new
// (rows x cols) dimensions.
PieceDistribution generate_distribution(const PieceDistribution& dist_4x10,
                                        int rows, int cols) {
    if (rows == 4 && cols == 10) {
        return dist_4x10;
    }

    PieceDistribution new_dist;

    for (const auto& [type, grid] : dist_4x10) {
        std::vector<std::vector<double>> new_grid(rows, std::vector<double>(cols));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Map target cell → source space
                double src_i = (i + 0.5) * 4.0 / rows - 0.5;
                double src_j = (j + 0.5) * 10.0 / cols - 0.5;

                int i0 = std::floor(src_i);
                int j0 = std::floor(src_j);
                int i1 = std::min(i0 + 1, 3);
                int j1 = std::min(j0 + 1, 9);

                double di = src_i - i0;
                double dj = src_j - j0;

                i0 = std::clamp(i0, 0, 3);
                j0 = std::clamp(j0, 0, 9);

                // Bilinear interpolation
                double v =
                    (1 - di) * (1 - dj) * grid[i0][j0] +
                    (1 - di) * dj       * grid[i0][j1] +
                    di       * (1 - dj) * grid[i1][j0] +
                    di       * dj       * grid[i1][j1];

                new_grid[i][j] = v;
            }
        }

        new_dist[type] = new_grid;
    }

    // Normalize every piece's distribution grid
    for (auto& [piece_type, grid] : new_dist) {
        normalize_grid(grid);
    }

    return new_dist;
}

void generate_probabilistic_setup(Board& board, Player player) {
    const auto& config = board.get_config();
    int rows = config.setup_rows;
    int cols = config.width;

    // 1. Load and aggregate distribution data
    PieceDistribution dist;
    try {
        // Attempt to load. In a production engine, this should be cached in memory 
        // rather than hitting the disk on every single setup.
        PieceDistribution raw_dist = load_4x10_distributions_from_json("data/setup_distributions.json");
        dist = generate_distribution(raw_dist, rows, cols);
    } catch (const std::exception& e) {
        // If file is missing, 'dist' remains empty. 
        // The algorithm below naturally falls back to uniform randomness.
    }

    // 2. Flatten army from counts to a placeable list
    std::vector<PieceType> army;
    for (const auto& [type, count] : config.piece_counts) {
        for (int i = 0; i < count; ++i) {
            army.push_back(type);
        }
    }

    // 3. Sort descending. 
    // This ensures Flags (12) and Bombs (11) get to pick their optimal spots 
    // first, before low-value pieces crowd them out.
    std::sort(army.begin(), army.end(), std::greater<PieceType>());

    // 4. Initialize available local slots [0, rows) x [0, cols)
    struct Slot { int r; int c; };
    std::vector<Slot> available_slots;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            available_slots.push_back({r, c});
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // 5. Iterative Weighted Sampling
    for (PieceType type : army) {
        std::vector<double> weights;
        weights.reserve(available_slots.size());
        
        bool has_dist = (dist.count(type) > 0);
        double total_weight = 0.0;

        // Build the probability weight array for the REMAINING available slots
        for (const auto& slot : available_slots) {
            double w = has_dist ? dist[type][slot.r][slot.c] : 1.0;
            weights.push_back(w);
            total_weight += w;
        }

        int chosen_index = 0;
        
        if (total_weight > 0.0) {
            // Pick a slot based on the parsed JSON probabilities
            std::discrete_distribution<int> discrete_dist(weights.begin(), weights.end());
            chosen_index = discrete_dist(gen);
        } else {
            // Fallback: If probabilities are 0 (or file missing), pick uniformly
            std::uniform_int_distribution<int> uniform_dist(0, available_slots.size() - 1);
            chosen_index = uniform_dist(gen);
        }

        Slot chosen_slot = available_slots[chosen_index];
        
        // Remove the chosen slot so no two pieces share it
        available_slots.erase(available_slots.begin() + chosen_index);

        // 6. Map the local setup slot to global board coordinates
        // Convention: local r=0 is the player's back row.
        int board_x = chosen_slot.c;
        int board_y = 0;

        if (player == Player::Blue) {
            board_y = chosen_slot.r; // Blue's back row is at y=0
        } else {
            board_y = config.height - 1 - chosen_slot.r; // Red's back row is at y=height-1
        }

        board.place_piece(board_x, board_y, type, player);
    }
}

void generate_random_setup(Board& board, Player player) {
    const auto& config = board.get_config();

    std::vector<PieceType> army;
    for (const auto& [type, count] : config.piece_counts) {
        for (int i = 0; i < count; ++i) {
            army.push_back(type);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(army.begin(), army.end(), gen);

    int start_y = (player == Player::Blue) ? 0 : config.height - config.setup_rows;
    int end_y = (player == Player::Blue) ? config.setup_rows : config.height;

    int idx = 0;
    for (int y = start_y; y < end_y; ++y) {
        for (int x = 0; x < config.width; ++x) {
            if (idx < army.size()) {
                board.place_piece(x, y, army[idx++], player);
            }
        }
    }
}

} // namespace stratego::setup