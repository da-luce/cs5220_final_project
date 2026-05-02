#include "state_convert.h"
#include <cstring>

namespace stratego_gpu {

void to_gpu(const stratego::GameState& cpu, GameStateGPU& gpu) {
    std::memset(&gpu, 0, sizeof(gpu));

    const auto& grid = cpu.board.grid;
    for (int i = 0; i < NUM_CELLS; ++i) {
        const auto& p = grid[i];

        gpu.board.types[i] = static_cast<uint8_t>(p.type);

        const uint64_t bit_lo = (i < 64) ? (1ull << i) : 0ull;
        const uint64_t bit_hi = (i < 64) ? 0ull : (1ull << (i - 64));

        if (p.owner == stratego::Player::Red) {
            gpu.board.occ_red[0] |= bit_lo;
            gpu.board.occ_red[1] |= bit_hi;
        } else if (p.owner == stratego::Player::Blue) {
            gpu.board.occ_blue[0] |= bit_lo;
            gpu.board.occ_blue[1] |= bit_hi;
        }
        if (p.type == stratego::PieceType::Water) {
            gpu.board.water[0] |= bit_lo;
            gpu.board.water[1] |= bit_hi;
        }
        if (p.revealed) {
            gpu.board.revealed[0] |= bit_lo;
            gpu.board.revealed[1] |= bit_hi;
        }
    }

    gpu.current_turn = (cpu.current_turn == stratego::Player::Red) ? PL_RED : PL_BLUE;

    // Derive the most recent move for each player from move_history.
    // CPU convention: state::initialize sets current_turn=Red, so move 0 is by Red,
    // move 1 by Blue, move 2 by Red, … (Red on even indices, Blue on odd).
    gpu.last_move[PL_RED]  = MoveGPU{LAST_MOVE_NONE, 0, 0, 0};
    gpu.last_move[PL_BLUE] = MoveGPU{LAST_MOVE_NONE, 0, 0, 0};
    const auto& mh = cpu.move_history;
    for (size_t i = 0; i < mh.size(); ++i) {
        const int player = (i % 2 == 0) ? PL_RED : PL_BLUE;
        const auto& m = mh[i];
        gpu.last_move[player] = MoveGPU{
            static_cast<uint8_t>(m.start_x), static_cast<uint8_t>(m.start_y),
            static_cast<uint8_t>(m.end_x),   static_cast<uint8_t>(m.end_y)
        };
    }
}

void from_gpu(const GameStateGPU& gpu, stratego::GameState& cpu) {
    auto& grid = cpu.board.grid;
    for (int i = 0; i < NUM_CELLS; ++i) {
        auto& p = grid[i];
        p.type = static_cast<stratego::PieceType>(gpu.board.types[i]);

        const bool red  = bm_test(i, gpu.board.occ_red[0],  gpu.board.occ_red[1]);
        const bool blue = bm_test(i, gpu.board.occ_blue[0], gpu.board.occ_blue[1]);
        if (red)        p.owner = stratego::Player::Red;
        else if (blue)  p.owner = stratego::Player::Blue;
        else            p.owner = stratego::Player::None;

        p.revealed = bm_test(i, gpu.board.revealed[0], gpu.board.revealed[1]);
    }

    cpu.current_turn = (gpu.current_turn == PL_RED) ? stratego::Player::Red
                                                   : stratego::Player::Blue;
}

} // namespace stratego_gpu
