#pragma once
#include "stratego.h"
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
PieceDistribution load_distributions_from_json(const std::string& filename);

// Generates a probabilistic piece setup for a given player based on expert data.
// This function implements an "Iterative Weighted Sampling" algorithm. For each
// piece to be placed, it samples an available slot based on the weighted
// probabilities from the distribution data, ensuring stronger pieces are more
// likely to appear in strategically sound locations.
void generate_probabilistic_setup(Board& board, Player player, const PieceDistribution& distributions);

} // namespace setup
} // namespace stratego