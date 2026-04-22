#pragma once
#include "stratego/board.h"
#include "stratego/state.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

/** Feature extraction scheme:
 *
 * Let N = number of piece types that can be placed (excluding Empty and Water)
 * We will have a total of (2N + 2) channels in our observation tensor:
 * [0 to N-1]   : My pieces (in order increasing by piece strength, e.g. Spy=0, Scout=1, ..., Marshal=10)
 * [N]          : Unrevealed opponent pieces
 * [N+1 to 2N]  : Revealed opponent pieces (also in increasing order by piece strength)
 * [2N+1]       : Obstacles (Water)
 *
 * We encode into a flat vector of floats for maximum compatibility. It is easy
 * to reshape this into a (Channels, Height, Width) tensor in Python using the
 * known board dimensions.
 * 
 * TODO: perhaps we should add last move channels, or other things?
 */
using BoardFeatures = std::vector<float>;

/**
 * Generates features from the perspective of perspective_player.
 * The board will be "normalized" (flipped) so that the perspective_player 
 * is always at the bottom of the H dimension.
 */
BoardFeatures get_board_encoding(const stratego::GameState& state, stratego::Player perspective_player);