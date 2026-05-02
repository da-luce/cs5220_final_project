#include "stratego_gpu.h"

namespace stratego_gpu {

// ─────────────────── 13×13 collision-outcome table ───────────────────
// Populated on the host and pushed to __constant__ memory by init_constants().
// Indices are attacker_type × defender_type, both PieceType IDs in [0,12].
//
//   Attacker     | Defender    | Result
//   -------------+-------------+-------------------
//   any (mobile) | flag        | CO_WIN_GAME
//   miner        | bomb        | CO_DEFENDER_DIES (defused)
//   not miner    | bomb        | CO_ATTACKER_DIES
//   spy          | marshal     | CO_DEFENDER_DIES (special rule)
//   rank a       | rank a      | CO_BOTH_DIE
//   rank a       | rank b<a    | CO_DEFENDER_DIES
//   rank a       | rank b>a    | CO_ATTACKER_DIES
//   anything else (illegal)    | CO_ILLEGAL = 0xFF
__constant__ uint8_t collision_table[NUM_PIECE_TYPES][NUM_PIECE_TYPES];

void init_constants() {
    uint8_t h[NUM_PIECE_TYPES][NUM_PIECE_TYPES];
    for (int a = 0; a < NUM_PIECE_TYPES; ++a)
        for (int d = 0; d < NUM_PIECE_TYPES; ++d)
            h[a][d] = CO_ILLEGAL;

    // Mobile attacker ranks are SPY..MARSHAL (1..10); only those reach combat.
    for (int a = PT_SPY; a <= PT_MARSHAL; ++a) {
        for (int d = PT_SPY; d <= PT_FLAG; ++d) {
            if (d == PT_FLAG) {
                h[a][d] = CO_WIN_GAME;
            } else if (d == PT_BOMB) {
                h[a][d] = (a == PT_MINER) ? CO_DEFENDER_DIES : CO_ATTACKER_DIES;
            } else if (a == PT_SPY && d == PT_MARSHAL) {
                h[a][d] = CO_DEFENDER_DIES;
            } else if (a == d) {
                h[a][d] = CO_BOTH_DIE;
            } else if (a > d) {
                h[a][d] = CO_DEFENDER_DIES;
            } else {
                h[a][d] = CO_ATTACKER_DIES;
            }
        }
    }
    cudaMemcpyToSymbol(collision_table, h, sizeof(h));
}

// ─────────────────── Bitmap helpers (device) ───────────────────
__device__ __forceinline__ void bm_set(uint64_t* bm, int idx) {
    if (idx < 64) bm[0] |= (1ull << idx);
    else          bm[1] |= (1ull << (idx - 64));
}

__device__ __forceinline__ void bm_clear(uint64_t* bm, int idx) {
    if (idx < 64) bm[0] &= ~(1ull << idx);
    else          bm[1] &= ~(1ull << (idx - 64));
}

// ─────────────────── Move generation kernel ───────────────────
//
// One block per game. Threads 0..99 are the source cells; 100..127 are inert.
// Every thread runs the SAME 4 × 9 direction × distance loop in lockstep — the
// `keep` predicate decides which thread's iteration produces a move. This is
// the design the user asked for: "every thread in the warp executes the
// instructions for all move types, but the results are only kept if they
// apply to the piece that the thread is currently responsible for."
//
// Repetition guard: also blocks any move that would immediately undo this
// player's previous move — a simplified two-squares rule. The previous move
// for `current_turn` is stashed in `GameStateGPU::last_move[current_turn]`,
// with start_x == LAST_MOVE_NONE meaning "no prior move".
__global__ void gen_legal_moves_kernel(
    const GameStateGPU* __restrict__ states,
    MoveGPU*            moves_out,
    int*                move_counts,
    int                 batch_size)
{
    const int game = blockIdx.x;
    const int tid  = threadIdx.x;
    if (game >= batch_size) return;

    // Cooperatively load the BoardGPU into shared memory.
    // sizeof(BoardGPU) = 4 × 16 (occ_red/blue/water/revealed) + 100 (types) = 164 bytes
    // = 41 uint32 words. We also stash current_turn, the per-block move counter
    // (kept in __shared__ to avoid serializing on a global atomic), and the
    // current player's last move for the reversal guard.
    __shared__ BoardGPU sb;
    __shared__ uint8_t  s_turn;
    __shared__ MoveGPU  s_last_move;
    __shared__ int      s_count;
    {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&states[game].board);
        uint32_t*       dst = reinterpret_cast<uint32_t*>(&sb);
        constexpr int WORDS = sizeof(BoardGPU) / 4;
        for (int i = tid; i < WORDS; i += blockDim.x) dst[i] = src[i];
    }
    if (tid == 0) {
        s_turn      = states[game].current_turn;
        s_last_move = states[game].last_move[states[game].current_turn];
        s_count     = 0;
    }
    __syncthreads();

    const int  src    = tid;
    const bool active = (src < NUM_CELLS);

    const int sx = active ? (src % BOARD_W) : 0;
    const int sy = active ? (src / BOARD_W) : 0;

    const uint8_t src_type = active ? sb.types[src] : (uint8_t)PT_EMPTY;

    const uint64_t my_lo = (s_turn == PL_RED) ? sb.occ_red[0]  : sb.occ_blue[0];
    const uint64_t my_hi = (s_turn == PL_RED) ? sb.occ_red[1]  : sb.occ_blue[1];

    const bool is_mine   = active && bm_test(src, my_lo, my_hi);
    const bool is_mobile = (src_type != PT_EMPTY && src_type != PT_WATER &&
                            src_type != PT_BOMB  && src_type != PT_FLAG);
    const bool valid_src = is_mine && is_mobile;
    const int  max_dist  = (src_type == PT_SCOUT) ? MAX_DIST : 1;

    // Reversal guard: if the current player previously moved a piece from A→B,
    // disallow a follow-up move B→A (the same piece undoing its last step).
    // The thread that owns square B (s_last_move.end) checks for moves to A.
    const bool last_valid = (s_last_move.start_x != LAST_MOVE_NONE);
    const bool is_last_dst = active && last_valid &&
                             (sx == (int)s_last_move.end_x) &&
                             (sy == (int)s_last_move.end_y);

    constexpr int DX[NUM_DIRS] = { 0,  0, -1,  1};
    constexpr int DY[NUM_DIRS] = {-1,  1,  0,  0};

    MoveGPU* my_moves_base = moves_out + game * MAX_MOVES_PER_STATE;

    // 4 dirs × 9 dists = 36 iterations; control flow is identical across the warp.
    #pragma unroll
    for (int dir = 0; dir < NUM_DIRS; ++dir) {
        // Inactive / immobile threads start "blocked" so their iterations are inert.
        bool blocked = !valid_src;

        #pragma unroll
        for (int dist = 1; dist <= MAX_DIST; ++dist) {
            const int nx = sx + DX[dir] * dist;
            const int ny = sy + DY[dir] * dist;

            const bool in_bounds = (nx >= 0 && nx < BOARD_W && ny >= 0 && ny < BOARD_H);
            const int  nidx      = in_bounds ? cell_idx(nx, ny) : 0;

            const uint8_t tgt_type = in_bounds ? sb.types[nidx] : (uint8_t)PT_WATER;
            const bool tgt_water   = (tgt_type == PT_WATER);
            const bool tgt_mine    = in_bounds && bm_test(nidx, my_lo, my_hi);
            const bool tgt_empty   = (tgt_type == PT_EMPTY);
            const bool tgt_enemy   = in_bounds && !tgt_empty && !tgt_water && !tgt_mine;

            const bool in_range = (dist <= max_dist);

            // Reversal guard: if this thread owns the previous move's end square,
            // forbid the candidate that lands on the previous move's start square.
            const bool breaks_two_square = is_last_dst &&
                                           (nx == (int)s_last_move.start_x) &&
                                           (ny == (int)s_last_move.start_y);

            // The "keep this move" predicate. Every thread evaluates it every iter;
            // only legal-for-this-piece moves get written.
            const bool keep = valid_src && in_range && in_bounds && !blocked &&
                              !tgt_water && !tgt_mine && !breaks_two_square;

            if (keep) {
                // Atomic on shared memory — much cheaper than the global address
                // 100 threads would otherwise contend on.
                int slot = atomicAdd(&s_count, 1);
                if (slot < MAX_MOVES_PER_STATE) {
                    MoveGPU m;
                    m.start_x = (uint8_t)sx;
                    m.start_y = (uint8_t)sy;
                    m.end_x   = (uint8_t)nx;
                    m.end_y   = (uint8_t)ny;
                    my_moves_base[slot] = m;
                }
            }

            // Once we hit anything non-empty (or step out of bounds) the rest of
            // this direction is unreachable — including for scouts past an enemy.
            if (!in_bounds || tgt_water || tgt_mine || tgt_enemy) blocked = true;
        }
    }

    __syncthreads();
    if (tid == 0) move_counts[game] = s_count;
}

void launch_gen_legal_moves(
    const GameStateGPU* d_states,
    MoveGPU*            d_moves_out,
    int*                d_move_counts,
    int                 batch_size,
    cudaStream_t        stream)
{
    constexpr int THREADS_PER_BLOCK = 128; // ceil(NUM_CELLS / 32) * 32 = 128
    gen_legal_moves_kernel<<<batch_size, THREADS_PER_BLOCK, 0, stream>>>(
        d_states, d_moves_out, d_move_counts, batch_size);
}

// ─────────────────── Move execution kernel ───────────────────
//
// One thread per game (cheap; the heavy work is move *generation*). Trusts the
// move is legal — typically you'd sample from the output of the generator.
__global__ void execute_move_kernel(
    GameStateGPU*  states,
    const MoveGPU* moves,
    uint8_t*       outcomes,
    int            batch_size)
{
    const int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch_size) return;

    GameStateGPU& s = states[game];
    const MoveGPU m = moves[game];

    const int s_idx = cell_idx(m.start_x, m.start_y);
    const int e_idx = cell_idx(m.end_x,   m.end_y);

    const uint8_t a_type = s.board.types[s_idx];
    const uint8_t d_type = s.board.types[e_idx];
    const uint8_t side   = s.current_turn;

    uint64_t* my_occ  = (side == PL_RED) ? s.board.occ_red  : s.board.occ_blue;
    uint64_t* opp_occ = (side == PL_RED) ? s.board.occ_blue : s.board.occ_red;

    // Snapshot the attacker's revealed bit so a quiet move can carry it along.
    const bool attacker_was_revealed =
        bm_test(s_idx, s.board.revealed[0], s.board.revealed[1]);

    uint8_t outcome = CO_ILLEGAL; // sentinel for "no combat" (quiet move)

    if (d_type == PT_EMPTY) {
        // Quiet move — piece (and its `revealed` flag) slides to the new cell.
        s.board.types[e_idx] = a_type;
        s.board.types[s_idx] = PT_EMPTY;
        bm_clear(my_occ,         s_idx);
        bm_set  (my_occ,         e_idx);
        bm_clear(s.board.revealed, s_idx);
        if (attacker_was_revealed) bm_set  (s.board.revealed, e_idx);
        else                       bm_clear(s.board.revealed, e_idx);
    } else {
        // Combat — outcome lookup in constant memory. Any survivor of combat is
        // "revealed" to both players (mirrors the CPU engine).
        outcome = collision_table[a_type][d_type];
        switch (outcome) {
            case CO_ATTACKER_DIES:
                // Defender survives at e_idx, revealed.
                s.board.types[s_idx] = PT_EMPTY;
                bm_clear(my_occ,         s_idx);
                bm_clear(s.board.revealed, s_idx);
                bm_set  (s.board.revealed, e_idx);
                break;
            case CO_DEFENDER_DIES:
            case CO_WIN_GAME:
                // Attacker survives, moves to e_idx, revealed.
                s.board.types[e_idx] = a_type;
                s.board.types[s_idx] = PT_EMPTY;
                bm_clear(my_occ,           s_idx);
                bm_clear(opp_occ,          e_idx);
                bm_set  (my_occ,           e_idx);
                bm_clear(s.board.revealed, s_idx);
                bm_set  (s.board.revealed, e_idx);
                break;
            case CO_BOTH_DIE:
                s.board.types[s_idx] = PT_EMPTY;
                s.board.types[e_idx] = PT_EMPTY;
                bm_clear(my_occ,           s_idx);
                bm_clear(opp_occ,          e_idx);
                bm_clear(s.board.revealed, s_idx);
                bm_clear(s.board.revealed, e_idx);
                break;
            default:
                // Should not happen for validated moves.
                break;
        }
    }

    // Record this player's most recent move for the next reversal guard.
    s.last_move[side] = m;

    s.current_turn   = (side == PL_RED) ? PL_BLUE : PL_RED;
    outcomes[game]   = outcome;
}

void launch_execute_move(
    GameStateGPU*  d_states,
    const MoveGPU* d_moves,
    uint8_t*       d_outcomes,
    int            batch_size,
    cudaStream_t   stream)
{
    constexpr int THREADS_PER_BLOCK = 128;
    int blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    execute_move_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_states, d_moves, d_outcomes, batch_size);
}

} // namespace stratego_gpu
