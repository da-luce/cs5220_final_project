# Stratego AI Bot API Specification (UCC 2012 Format)

In 2012, the The University Computer Club Inc. of the University of Western Australia hosted a Stratego bot competition. This document describes the standard input/output (stdio) text protocol used by the game manager to communicate with AI bots. Bots act as standalone processes. The game manager communicates with bots by writing to their `stdin` and reading their moves from `stdout`.

The Stratego AI Evaluator is a manager program used to evaluate the strength of ai bots used to play the game of Stratego, a two player game in the capture-the-flag genre. It was developed by Sam Moore of the University Computer Club in Australia to judge a 2012 programming competition where several stratego AI bots competed. It's source is added a a submodule in [third_party/strategoevaluator](./third_party/strategoevaluator).

---

## 1. Important: Disable Buffering
Because the communication is synchronous and turn-based over standard streams, bots **must disable output buffering** to prevent the manager from hanging while waiting for a move.
* **C++**: `cin.rdbuf()->pubsetbuf(NULL, 0); cout.rdbuf()->pubsetbuf(NULL, 0);`
* **Python**: Run python with the `-u` flag (`python -u bot.py`) or use `sys.stdout.flush()`.
* **C**: `setvbuf(stdout, NULL, _IONBF, 0);`

---

## 2. Setup Phase

When the manager launches the bot, it sends the initial game parameters.

### Input from Manager (Read via `stdin`):

```text
<COLOUR>
<OPPONENT_NAME>
<WIDTH> <HEIGHT>
<COLOUR>: RED or BLUE.
<OPPONENT_NAME>: String representing the opponent's name.
<WIDTH>: Board width (typically 10).
<HEIGHT>: Board height (typically 10).
```

### Output from Bot (Write to stdout):

The bot must immediately output its starting formation. For a standard 10x10 game, the setup zone is `10x4`.

The output must be 4 lines of 10 characters each.

Piece Characters:

- `F` (Flag), `B` (Bomb), `s` (Spy)
- `1` through `9` (Ranks: 1=Marshal, 2=General, ..., 8=Miner, 9=Scout)

IMPORTANT: notice that this uses the American style for piece numbering (lower numbers are more powerful), but our program uses the European convention.

Example Output:

```text
FB8sB479B8
BB31555583
6724898974
967B669999
```

---

## 3. Initial Board State (Turn 0 Only)

Immediately after providing the setup, the manager sends the full board state to the bot. The bot reads `<HEIGHT>` number of lines, each containing `<WIDTH>` characters.

Map Characters:

- `.` : Empty space
- `+` : Impassable terrain (Water/Boulder)
- `#` : Unknown enemy unit
- `[A-Za-z0-9]` : Your own units (matching the characters you provided in setup)

---

## 4. Game Loop

After Turn 0, the game enters a continuous loop of **Result Interpretation -> Move Generation.**

#### A. Read the Result of the Previous Turn (stdin)

The manager tells the bot the outcome of the last action (either the bot's own action, or the opponent's action).

Format:

```text
<X> <Y> <DIRECTION> <MULTIPLIER> <OUTCOME> [EXTRA_INFO]
```

- **X**, **Y**: The starting coordinates of the piece that moved.
- DIRECTION: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- MULTIPLIER: Distance moved (usually 1, but Scouts can move further).
- OUTCOME: What happened as a result of the move.

Possible Outcomes:

- `OK`: The piece moved to an empty square safely.
- `KILLS <RANK>`: The moving piece attacked and won. `<RANK>` is the token of the defeated defender.
- `DIES <RANK>`: The moving piece attacked and lost. `<RANK>` is the token of the victorious defender.
- `BOTHDIE`: Both pieces had the same rank and destroyed each other.
- `NO_MOVE`: The opponent passed or had no valid moves.
- `FLAG`: The flag was captured. The game is over.
- `ILLEGAL`: An illegal move was attempted. The game is over.
- `QUIT`: The manager is terminating the game/process. The bot should exit.

#### B. Output Your Move (stdout)

When it is the bot's turn, it must output a single line describing its desired move: Plaintext

`<X> <Y> <DIRECTION> [MULTIPLIER]`

- Example: `3 4 UP`
- If the bot has no mobile pieces left, it must output: `NO_MOVE`