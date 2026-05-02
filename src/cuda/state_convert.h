#pragma once
//
// Host-side bridge between the CPU `stratego::GameState` and the GPU
// `stratego_gpu::GameStateGPU`. Pure host code — safe to include in .cpp.
//

#include "stratego_gpu.h"
#include "stratego/state.h"

namespace stratego_gpu {

// Pack a CPU GameState into a GameStateGPU (water/owner bitmaps + per-cell types).
void to_gpu(const stratego::GameState& cpu, GameStateGPU& gpu);

// Unpack a GameStateGPU back into the `board.grid` of `cpu` (and current_turn).
// `cpu` must already have a board with the right config so we know the geometry.
void from_gpu(const GameStateGPU& gpu, stratego::GameState& cpu);

} // namespace stratego_gpu
