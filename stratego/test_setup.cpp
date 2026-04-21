#include <gtest/gtest.h>
#include "setup.h"
#include <vector>

using namespace stratego;
using namespace stratego::setup;

// Helper to mathematically verify a 2D probability grid using GTest macros
void ExpectValidDistribution(const std::vector<std::vector<double>>& grid) {
    double sum = 0.0;
    
    for (const auto& row : grid) {
        for (double p : row) {
            // Probabilities must be between 0 and 1
            EXPECT_GE(p, 0.0) << "Probability fell below 0.0";
            EXPECT_LE(p, 1.0) << "Probability exceeded 1.0";
            sum += p;
        }
    }
    
    // Probabilities must sum exactly to 1.0 (with a 1e-6 margin for float error)
    EXPECT_NEAR(sum, 1.0, 1e-6) << "Distribution does not sum to 1.0";
}

TEST(SetupGeneratorTest, DistributionResizingAndMath) {
    // 1. Create a dummy 4x10 source distribution
    PieceDistribution mock_dist_4x10;
    
    // Create a perfectly uniform distribution for the Flag (1/40 = 0.025 per cell)
    std::vector<std::vector<double>> uniform_grid(4, std::vector<double>(10, 0.025));
    mock_dist_4x10[PieceType::Flag] = uniform_grid;

    // Create a highly skewed distribution for the Bomb (1.0 in a single cell)
    std::vector<std::vector<double>> skewed_grid(4, std::vector<double>(10, 0.0));
    skewed_grid[0][0] = 1.0; 
    mock_dist_4x10[PieceType::Bomb] = skewed_grid;

    // 2. Test generating a Tiny Stratego distribution (3x8)
    PieceDistribution dist_3x8 = generate_distribution(mock_dist_4x10, 3, 8);
    
    EXPECT_GT(dist_3x8.count(PieceType::Flag), 0);
    EXPECT_EQ(dist_3x8[PieceType::Flag].size(), 3);
    EXPECT_EQ(dist_3x8[PieceType::Flag][0].size(), 8);
    
    ExpectValidDistribution(dist_3x8[PieceType::Flag]);
    ExpectValidDistribution(dist_3x8[PieceType::Bomb]);

    // 3. Test generating a Quick Stratego distribution (4x10 - identical bypass)
    PieceDistribution dist_4x10 = generate_distribution(mock_dist_4x10, 4, 10);
    ExpectValidDistribution(dist_4x10[PieceType::Flag]);
    ExpectValidDistribution(dist_4x10[PieceType::Bomb]);

    // 4. Test extreme compression (e.g., 2x5 half-board)
    PieceDistribution dist_2x5 = generate_distribution(mock_dist_4x10, 2, 5);
    EXPECT_EQ(dist_2x5[PieceType::Flag].size(), 2);
    EXPECT_EQ(dist_2x5[PieceType::Flag][0].size(), 5);
    ExpectValidDistribution(dist_2x5[PieceType::Flag]);
    ExpectValidDistribution(dist_2x5[PieceType::Bomb]);
}