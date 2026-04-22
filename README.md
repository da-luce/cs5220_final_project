# CS 5220 SP2026 Final Project: Scaling Parallel Stratego Training

**Cornell University** \
**Authors:** [Abrar Amin](https://github.com/abrar-amin) & [Dalton Luce](https://github.com/da-luce)

- [CS 5220 SP 2026](https://www.cs.cornell.edu/courses/cs5220/2026sp/)
- [CS 5782 SP2026](https://www.cs.cornell.edu/courses/cs4782/2026sp/): helpful RL background resources

## Overview

This project explores the scalability and stability tradeoffs of parallelized Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) via self-play in the game of [Stratego](https://en.wikipedia.org/wiki/Stratego). By distributing environment simulation and training across multiple workers, we aim to maximize experience generation throughput.

To manage computational complexity, training begins on [Stratego Tiny](https://www.diva-portal.org/smash/get/diva2:1656447/FULLTEXT01.pdf#section.2.1) (a reduced state/action space variant) and will progressively scale toward the full board state.

We use the modern Stratego numbering system, where larger numbers indicate higher piece value. We do not use the Spotter piece for simplicity. We provide three game types:

- Classic: 10x10 grid, 40 pieces
- Barrage: 10x10 grid, 8 pieces
- Quick: 8x8 grid, 10 pieces
- Tiny: 6x6 grid, 6 pieces

When playing Classic, the starting setup is sampled from a Constrained Weighted Distribution based on the Dobby/Oewesok marginals (see [data](./data/)). The placement logic follows an iterative masking approach: pieces are sampled sequentially, starting with the Flag and Bombs, with the probability map re-normalized after each tile is occupied. This ensures valid, non-overlapping configurations that mirror human expert heuristics while maintaining high entropy for RL training. In practice, these setups are not ideal, as sampling from a distribution for each piece individually doesn't consider the synergy of piece clusters, such as the critical proximity of the Spy to the Marshal or the mutual protection of Bomb-Flag formations. However, it is much better than complete randomness.


## Building

```shell
mkdir build
cd build
cmake ..
make
```

### Run Tests

```shell
./stratego/test_stratego
```

## Run Terminal Game

```shell
./frontend/play
```

## Training

```shell
./rl/train --iters 50000 --out custom_best_model.pth
```

# TODO: Training on Perlmutter

## Architecture & Tech Stack

This project is built for high-performance execution on the [Perlmutter](https://www.nersc.gov/what-we-do/computing-for-science/perlmutter) supercomputer. To minimize overhead, the entire pipeline is a pure C++ implementation.

* **Algorithm:** Proximal Policy Optimization (PPO) (Actor-Critic)
* **Environment:** Custom C++ Stratego grid simulation
* **Distributed Communication:** NCCL (NVIDIA Collective Communications Library) 
* **Target Hardware:** Perlmutter (Multi-GPU node scaling)

## Core Interfaces

As opposed to directly following established RL frameworks like OpenSpiel, we opted for a minimal set of simple abstractions tailored to our needs. The system is decoupled into three primary C++ interfaces to allow for rapid iteration and testing:
1. **`Environment`:** A Gymnasium-style interface handling `reset`, `step`, and game logic (imperfect information masking, combat resolution, 500-step truncation).
2. **`Agent`:** A dual-headed Actor-Critic model. It outputs actions/log-probs for self-play rollouts and evaluates value estimations for the PPO update.
3. **`RolloutBuffer`:** An on-policy data store that collects step trajectories and computes Generalized Advantage Estimation (GAE) before flushing.

## Training Pipeline (Self-Play)

1. **Parallel Rollouts:** Multiple workers independently simulate games against a uniformly sampled pool of past checkpoints.
2. **Synchronization:** Gradients and model parameters are synchronized across GPUs using NCCL. We are evaluating both synchronous (stable but bottlenecked) and asynchronous (higher throughput but noisier) update strategies.
3. **Optimization:** The PPO clipped surrogate loss is calculated, and weights are updated via mini-batches.

```text
final_project/
├── CMakeLists.txt
├── README.md
├── main.cpp                  # Only parses args and launches train/eval modes
│
├── stratego/                 # PURE GAME LOGIC (No RL code here)
│   ├── stratego.h            # Board, Piece, Move definitions
│   └── stratego.cpp          # Move execution, combat resolution
│
├── environment/              # GYM-STYLE WRAPPERS
│   ├── environment.h         # Base Environment interface
│   └── stratego_env.cpp      # StrategoEnvironment class (generates onbs)
│
├── rl/                       # PPO & DISTRIBUTED TRAINING CODE
│   ├── agent.h               # Agent interface and struct definitions
│   ├── rollout_buffer.cpp    # Trajectory storage and GAE math
│   ├── ppo_trainer.cpp       # Training PPO
│   └── nccl_utils.cpp        # (Future) Perlmutter multi-GPU sync helpers
│
├── frontend/                 # VISUALIZATION & PLAY
│   └── play_ascii.cpp        # Your ncurses terminal UI
│
└── experiments/              # SCRIPTS & PROTOTYPES
    └── play.py               # Tkinter Python UI
```

## Random Notes and Observations

- Stratego logic is very conditional (e.g., if–then–else rules, piece rank comparisons), making it very hard to model or run quickly on a GPU
- Encoding board state is important

## Board Encoding

We have found that devising a smart encoding of the board state is critical to obtain good AI performance. Our scheme follows the standard approach used in modern deep reinforcement learning systems for board games, where the state is encoded as a stack of spatial feature planes separating piece types, player ownership, and hidden information. More specifically, our board state is represented as a `C x H x W` tensor, where `C` is the number of channels (see below), and `H x W` are the height and width of the game board, respectively. Each channel contains different information:

- For each piece ranking (Scout, Miner, etc.), encode a one hot channel for all piece locations.
-For each piece ranking (Flag, Lieutenant, Captain, Major), encode a one-hot channel for the current player's pieces
- Also, for each piece ranking, encode a one-hot channel for the opponent's pieces that have been revealed during combat
- Encode one channel for all opponent pieces whose identities are not yet revealed
- Encode one channel containing the lake (impassable tiles)

## Action Encoding

We use a flattened discrete action space where each possible move is represented as a single integer. This avoids structured outputs like `(x, y, dx, dy)` and allows the policy network to output a single categorical distribution over all actions. This is also standard in modern RL.

Each action encodes:

- a start position `(x, y)`
- a direction in the set  `{Up, Down, Left, Right}`
- a movement distance in the range `[1, max_dist]`, where `max_dist = max(H, W) - 1`

We flatten this into a single index:

$$
\text{action} = ((y \cdot W + x) \cdot 4 \cdot D) + (\text{direction} \cdot D) + (\text{distance} - 1)
$$

where

- $W$ = board width
- $H$ = board height
- $D$ = max distance
- $\text{dir} \in \{0,1,2,3\}$

This gives a total action space size:

$$
|A| = H \cdot W \cdot 4 \cdot D
$$

This gives us a fixed-size action space directly compatible with softmax policy heads. for a classic game of Stratego (H = 10, W = 10, D = 10), our action space has a size of 4000. This mapping is fully reversible so we can decode the action index back into an actual move on the game board. Illegal actions are handled via masking or penalization during environment stepping.