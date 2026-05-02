#pragma once
//
// GPU Stratego (Classic 10×10) — header
// =====================================
//
// Compact device-side representation of a Stratego position plus kernels
// that generate legal moves and apply moves entirely on the GPU. The aim is to
// keep an entire RL self-play loop resident on the device.
//
// Encoding choices
// ----------------
//  * 100 cells fit in a 2-word bitmap (bit i lives in word i/64, bit i%64).
//  * Per-cell piece type is stored as a uint8 in `BoardGPU::types[i]`.
//  * Combat outcomes are looked up in a 13×13 table held in __constant__ memory.
//
// Kernel layout
// -------------
//  * One CUDA block per game.
//  * 128 threads per block — threads 0..99 each "own" one source cell, threads
//    100..127 are idle. Every active thread executes the same 4×9 direction-
//    distance loop (Stratego has 4 cardinal directions, scouts can slide up to
//    9 squares); divergence is hidden by the `keep` predicate so the warp never
//    branches on per-piece logic.

#include <cuda_runtime.h>
#include <cstdint>

namespace stratego_gpu {

// ───── Board geometry ─────
constexpr int BOARD_W            = 10;
constexpr int BOARD_H            = 10;
constexpr int NUM_CELLS          = BOARD_W * BOARD_H;       // 100
constexpr int MAX_DIST           = 9;                       // scout max range (board width − 1)
constexpr int NUM_DIRS           = 4;                       // N, S, W, E
constexpr int MAX_MOVES_PER_STATE = 200;                    // generous upper bound

// ───── Piece type IDs (must match stratego::PieceType so static_cast works) ─────
enum PieceType : uint8_t {
    PT_EMPTY = 0, PT_SPY = 1, PT_SCOUT = 2, PT_MINER = 3,
    PT_SERGEANT = 4, PT_LIEUTENANT = 5, PT_CAPTAIN = 6, PT_MAJOR = 7,
    PT_COLONEL = 8, PT_GENERAL = 9, PT_MARSHAL = 10,
    PT_BOMB = 11, PT_FLAG = 12, PT_WATER = 13
};
constexpr int NUM_PIECE_TYPES = 13; // EMPTY..FLAG (water is never a defender)

// ───── Player IDs ─────
enum Player : uint8_t {
    PL_RED  = 0,
    PL_BLUE = 1,
    PL_NONE = 2,
};

// ───── Collision outcomes ─────
enum CollisionOutcome : uint8_t {
    CO_ATTACKER_DIES = 0, // attacker's rank loses
    CO_DEFENDER_DIES = 1, // defender's rank loses (Miner-vs-Bomb counts here)
    CO_BOTH_DIE      = 2, // equal ranks
    CO_WIN_GAME      = 3, // flag captured
    CO_ILLEGAL       = 0xFF, // not a real combat (e.g. moved to empty)
};

// 13×13 attacker-vs-defender lookup, populated by `init_constants()` and copied
// to constant memory. Defined in stratego_gpu.cu.
extern __constant__ uint8_t collision_table[NUM_PIECE_TYPES][NUM_PIECE_TYPES];

// ───── Board encoding ─────
struct BoardGPU {
    uint64_t occ_red[2];   // bit i <-> cell i is owned by red
    uint64_t occ_blue[2];  // bit i <-> cell i is owned by blue
    uint64_t water[2];     // bit i <-> cell i is a lake (constant for classic)
    uint64_t revealed[2];  // bit i <-> piece at cell i is revealed to both players
    uint8_t  types[NUM_CELLS]; // piece-type at cell i; PT_EMPTY for empty squares
};

struct MoveGPU {
    uint8_t start_x, start_y, end_x, end_y;
};

// Sentinel encoded in last_move when no prior move exists for that player.
// Picked because no real move has start_x = 0xFF on a 10×10 board.
constexpr uint8_t LAST_MOVE_NONE = 0xFF;

struct GameStateGPU {
    BoardGPU board;
    uint8_t  current_turn;          // PL_RED or PL_BLUE
    uint8_t  _pad[3];               // align last_move on a 4-byte boundary
    MoveGPU  last_move[2];          // last_move[PL_RED], last_move[PL_BLUE];
                                    // start_x == LAST_MOVE_NONE means "no prior move"
};

// ───── Helpers ─────
__host__ __device__ __forceinline__ int cell_idx(int x, int y) {
    return y * BOARD_W + x;
}

// Test bit `idx` in a (lo64, hi64) split bitmap representation of a 100-cell mask.
__host__ __device__ __forceinline__ bool bm_test(int idx, uint64_t lo, uint64_t hi) {
    return (idx < 64)
        ? ((lo >> idx) & 1ull) != 0
        : ((hi >> (idx - 64)) & 1ull) != 0;
}

// ───── Public API ─────

// Call once on the host before launching kernels. Copies the precomputed
// collision-outcome table into __constant__ memory.
void init_constants();

// Launch the legal-move generator. `d_states[i]` produces moves into
// `d_moves_out[i*MAX_MOVES_PER_STATE..]`, with the count written to
// `d_move_counts[i]`. Caller must zero/allocate buffers; the kernel resets
// each per-game count to 0 internally.
void launch_gen_legal_moves(
    const GameStateGPU* d_states,
    MoveGPU*            d_moves_out,
    int*                d_move_counts,
    int                 batch_size,
    cudaStream_t        stream = 0);

// Apply one move per game in-place. Assumes each move is legal (e.g. drawn from
// the output of `launch_gen_legal_moves`). `d_outcomes[i]` receives the combat
// outcome (or CO_ILLEGAL=0xFF when the move was a quiet move to an empty square).
void launch_execute_move(
    GameStateGPU*       d_states,
    const MoveGPU*      d_moves,
    uint8_t*            d_outcomes,
    int                 batch_size,
    cudaStream_t        stream = 0);

} // namespace stratego_gpu
