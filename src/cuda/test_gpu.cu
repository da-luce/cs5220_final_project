//
// CPU / GPU correctness check for legal-move generation.
// =====================================================
//
// For a batch of randomly initialized classic Stratego positions:
//   1. Convert each CPU state into the GPU representation.
//   2. Run the GPU kernel to generate legal moves.
//   3. Run the CPU `Engine::get_all_legal_moves` for the same state.
//   4. Sort both move sets and compare; any mismatch is reported with full detail.
//
// Notes on rule scope
// -------------------
// The GPU kernel currently does NOT enforce the chase rule (More-Squares) or the
// two-squares rule. Both rules only kick in when there is a non-empty move
// history. We therefore restrict comparison to fresh states — same scope where
// both implementations agree.
//
// To extend coverage to mid-game positions later, either (a) add the rules to
// the kernel, or (b) clear move_history / chase_hashes before calling the CPU
// engine in the comparison.

#include "stratego_gpu.h"
#include "state_convert.h"
#include "stratego/engine.h"
#include "stratego/state.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace sg = stratego_gpu;

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                     __FILE__, __LINE__, cudaGetErrorString(_err));            \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct MoveKey {
    int sx, sy, ex, ey;
    bool operator<(const MoveKey& o) const {
        return std::tie(sx, sy, ex, ey) < std::tie(o.sx, o.sy, o.ex, o.ey);
    }
    bool operator==(const MoveKey& o) const {
        return sx == o.sx && sy == o.sy && ex == o.ex && ey == o.ey;
    }
};

static std::vector<MoveKey> sorted_keys_cpu(const std::vector<stratego::Move>& v) {
    std::vector<MoveKey> k; k.reserve(v.size());
    for (const auto& m : v) k.push_back({m.start_x, m.start_y, m.end_x, m.end_y});
    std::sort(k.begin(), k.end());
    return k;
}

static std::vector<MoveKey> sorted_keys_gpu(const sg::MoveGPU* a, int n) {
    std::vector<MoveKey> k; k.reserve(n);
    for (int i = 0; i < n; ++i) k.push_back({a[i].start_x, a[i].start_y, a[i].end_x, a[i].end_y});
    std::sort(k.begin(), k.end());
    return k;
}

static void print_diff(int idx,
                       const std::vector<MoveKey>& cpu,
                       const std::vector<MoveKey>& gpu)
{
    std::printf("[FAIL] state %d: CPU=%zu moves, GPU=%zu moves\n",
                idx, cpu.size(), gpu.size());

    // CPU moves missing on GPU
    size_t i = 0, j = 0;
    while (i < cpu.size() && j < gpu.size()) {
        if (cpu[i] == gpu[j]) { ++i; ++j; continue; }
        if (cpu[i] < gpu[j]) {
            std::printf("    only CPU: (%d,%d)->(%d,%d)\n",
                        cpu[i].sx, cpu[i].sy, cpu[i].ex, cpu[i].ey);
            ++i;
        } else {
            std::printf("    only GPU: (%d,%d)->(%d,%d)\n",
                        gpu[j].sx, gpu[j].sy, gpu[j].ex, gpu[j].ey);
            ++j;
        }
    }
    for (; i < cpu.size(); ++i) std::printf("    only CPU: (%d,%d)->(%d,%d)\n",
                                            cpu[i].sx, cpu[i].sy, cpu[i].ex, cpu[i].ey);
    for (; j < gpu.size(); ++j) std::printf("    only GPU: (%d,%d)->(%d,%d)\n",
                                            gpu[j].sx, gpu[j].sy, gpu[j].ex, gpu[j].ey);
}

int main(int argc, char** argv) {
    int batch = (argc > 1) ? std::atoi(argv[1]) : 64;
    if (batch <= 0) batch = 64;

    sg::init_constants();
    CUDA_CHECK(cudaPeekAtLastError());

    // ── 1. Build CPU states ──
    std::vector<stratego::GameState> cpu_states;
    cpu_states.reserve(batch);
    for (int i = 0; i < batch; ++i) {
        cpu_states.push_back(stratego::state::initialize(
            stratego::get_config_for_game_type(stratego::GameType::Classic),
            stratego::state::SetupType::Random,
            /*max_moves=*/2000));
    }

    // ── 2. Convert and upload ──
    std::vector<sg::GameStateGPU> h_states(batch);
    for (int i = 0; i < batch; ++i) sg::to_gpu(cpu_states[i], h_states[i]);

    sg::GameStateGPU* d_states  = nullptr;
    sg::MoveGPU*      d_moves   = nullptr;
    int*              d_counts  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, batch * sizeof(*d_states)));
    CUDA_CHECK(cudaMalloc(&d_moves,  batch * sg::MAX_MOVES_PER_STATE * sizeof(*d_moves)));
    CUDA_CHECK(cudaMalloc(&d_counts, batch * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_states, h_states.data(),
                          batch * sizeof(sg::GameStateGPU),
                          cudaMemcpyHostToDevice));

    // ── 3. Launch the kernel ──
    sg::launch_gen_legal_moves(d_states, d_moves, d_counts, batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // ── 4. Pull results back ──
    std::vector<int>            h_counts(batch);
    std::vector<sg::MoveGPU>    h_moves(batch * sg::MAX_MOVES_PER_STATE);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                          batch * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_moves.data(), d_moves,
                          batch * sg::MAX_MOVES_PER_STATE * sizeof(sg::MoveGPU),
                          cudaMemcpyDeviceToHost));

    // ── 5. Compare ──
    int passed = 0, failed = 0;
    for (int i = 0; i < batch; ++i) {
        const auto cpu_moves = stratego::Engine::get_all_legal_moves(
            cpu_states[i], cpu_states[i].current_turn);

        const sg::MoveGPU* gpu_base = h_moves.data() + i * sg::MAX_MOVES_PER_STATE;
        const int gpu_n = h_counts[i];

        if (gpu_n > sg::MAX_MOVES_PER_STATE) {
            std::printf("[FAIL] state %d: GPU reported %d moves > MAX_MOVES_PER_STATE=%d\n",
                        i, gpu_n, sg::MAX_MOVES_PER_STATE);
            ++failed;
            continue;
        }

        auto cpu_keys = sorted_keys_cpu(cpu_moves);
        auto gpu_keys = sorted_keys_gpu(gpu_base, gpu_n);
        if (cpu_keys == gpu_keys) {
            ++passed;
        } else {
            ++failed;
            print_diff(i, cpu_keys, gpu_keys);
        }
    }

    std::printf("Result: %d / %d states matched.\n", passed, batch);

    cudaFree(d_states);
    cudaFree(d_moves);
    cudaFree(d_counts);
    return (failed == 0) ? 0 : 1;
}
