#pragma once
#include "board.h"
#include <string>
#include <map>
#include <vector>

namespace stratego {

// Forward declaration
class Board;

namespace setup {

// A data structure to hold the loaded probability distributions.
// The structure is a map from PieceType to a 4x10 grid of probabilities.
using PieceDistribution = std::map<PieceType, std::vector<std::vector<double>>>;

// Loads piece placement distributions from the provided JSON file.
// This defines a probability for each piece type to be placed in each of the
// 40 possible slots in the player's setup area on a standard 10x10 board. (4 setup rows)
PieceDistribution load_4x10_distributions_from_json(const std::string& filename);

// For other game modes, we can generate distributions by aggregating the 4x10
// data. The new game dimension is (rows x cols), and rows must be <= 4 and cols
// must be <= 10.
PieceDistribution generate_distribution(const PieceDistribution& dist_4x10, int rows, int cols);

// Generates a probabilistic piece setup for a given player based on expert data.
// This function implements an "Iterative Weighted Sampling" algorithm. For each
// piece to be placed, it samples an available slot based on the weighted
// probabilities from the distribution data, ensuring stronger pieces are more
// likely to appear in strategically sound locations.
void generate_probabilistic_setup(Board& board, Player player);

// Generates a completely random, valid piece setup for a given player.
void generate_random_setup(Board& board, Player player);

} // namespace setup
} // namespace stratego