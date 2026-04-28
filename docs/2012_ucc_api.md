# Stratego AI Bot API Specification (UCC 2012 Format)

In 2012, the The University Computer Club Inc. of the University of Western Australia hosted a Stratego bot **competition**. This document describes the standard input/output (stdio) text protocol used by the game manager to communicate with AI bots. Bots act as standalone processes. The game manager communicates with bots by writing to their `stdin` and reading their moves from `stdout`. It was developed by Sam Moore of UCC. It's source is added a a submodule in [third_party/strategoevaluator](./third_party/strategoevaluator).

## Coordinates

The UCC 2012 Stratego bots use an absolute global coordinate system. The Game Manager does not translate or flip the board based on which side you are playing. Your bot must be aware of its assigned color and handle its own spatial logic accordingly.

When programming your bot, adhere strictly to these absolute rules:

-  **The Origin:** The coordinate (0,0) is always the absolute top-left corner of the board. The X-axis ranges from 0 to 9 (left to right), and the Y-axis ranges from 0 to 9 (top to bottom).
- **The Starting Zones:** 
  - `RED` always starts at the top of the board. It must place its pieces on Rows 0, 1, 2, and 3
  - `BLUE` always starts at the bottom of the board. It must place its pieces on Rows 6, 7, 8, and 9
- **Movement Directions:** Directions are tied to the global grid, not your bot's forward-facing perspective.
  - `UP` always decreases the Y-coordinate (moving towards Row 0 / Red's side).
  - `DOWN` always increases the Y-coordinate (moving towards Row 9 / Blue's side).

## Special Notes

- The UCC bots do not adhere to the two-square or more-square moving rules, briefly described on [Wikipedia](https://en.wikipedia.org/wiki/Stratego#Rules_of_movement) and in more detail on this archived page for the [2010 Computer Stratego World Championship](https://web.archive.org/web/20110123114925/http://www.strategousa.org/wiki/index.php/2010_Computer_Stratego_World_Championship). Thus, they will occasionally make moves that violate these rules, resulting in a win for the other player.
- The `peternlewis` UCC bot has been observed to try and move non-scout pieces exactly 2 squares at a time when they lie on row 0. More investigation is needed as to why this happens.

---

## Protocol

### 1. Important: Disable Buffering
Because the communication is synchronous and turn-based over standard streams, bots **must disable output buffering** to prevent the manager from hanging while waiting for a move.
* **C++**: `cin.rdbuf()->pubsetbuf(NULL, 0); cout.rdbuf()->pubsetbuf(NULL, 0);`
* **Python**: Run python with the `-u` flag (`python -u bot.py`) or use `sys.stdout.flush()`.
* **C**: `setvbuf(stdout, NULL, _IONBF, 0);`

---

### 2. Setup Phase

When the manager launches the bot, it sends the initial game parameters.

#### Input from Manager (Read via `stdin`):

```text
<COLOUR> <OPPONENT_NAME> <WIDTH> <HEIGHT>
```

NOTE: These four parameters are sent on a SINGLE space-separated line. Bots using `getline()` must parse the entire line at once; attempting to read these as separate lines will result in a token count error and process failure.

- <COLOUR>: `RED` or `BLUE`
- <OPPONENT_NAME>: String representing the opponent's name.
- <WIDTH>: Board width (typically 10).
- <HEIGHT>: Board height (typically 10).

#### Output from Bot (Write to stdout):

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

### 3. Initial Board State (Turn 0 Only)

Immediately after providing the setup, the manager sends the full board state to the bot. The bot reads `<HEIGHT>` number of lines, each containing `<WIDTH>` characters.

Map Characters:

- `.` : Empty space
- `+` : Impassable terrain (Water/Boulder)
- `#` : Unknown enemy unit
- `[A-Za-z0-9]` : Your own units (matching the characters you provided in setup)

---

### 4. Game Loop

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


**CRITICAL TURN 0 BEHAVIOR:** > Every bot is programmed to discard exactly one line of text before reading the board for the first time.

- RED (Moves first) will receive the string `START`.
- BLUE (Moves second) will receive RED's first move (e.g., `5 6 UP 1 OK`). Ensure your loop logic consumes this "previous result" line even on the very first turn to stay in sync with the board state.

#### B. Output Your Move (stdout)

When it is the bot's turn, it must output a single line describing its desired move: Plaintext

`<X> <Y> <DIRECTION> [MULTIPLIER]`

- Example: `3 4 UP`
- If the bot has no mobile pieces left, it must output: `NO_MOVE`

### Example

```text
=== SETUP PHASE ===
MANAGER -> RED: RED
MANAGER -> RED: Blue_Destroyer
MANAGER -> RED: 10 10
RED -> MANAGER: FB8sB479B8
RED -> MANAGER: BB31555583
RED -> MANAGER: 6724898974
RED -> MANAGER: 967B669999

MANAGER -> BLUE: BLUE
MANAGER -> BLUE: Red_Menace
MANAGER -> BLUE: 10 10
BLUE -> MANAGER: 967B669999
BLUE -> MANAGER: 6724898974
BLUE -> MANAGER: BB31555583
BLUE -> MANAGER: FB8sB479B8


=== TURN 0: RED MOVES ===
MANAGER -> RED: ##########
MANAGER -> RED: ##########
MANAGER -> RED: ##########
MANAGER -> RED: ##########
MANAGER -> RED: ..++..++..
MANAGER -> RED: ..++..++..
MANAGER -> RED: 967B669999
MANAGER -> RED: 6724898974
MANAGER -> RED: BB31555583
MANAGER -> RED: FB8sB479B8
RED -> MANAGER: 0 6 UP


=== TURN 0: BLUE MOVES ===
MANAGER -> BLUE: 967B669999
MANAGER -> BLUE: 6724898974
MANAGER -> BLUE: BB31555583
MANAGER -> BLUE: FB8sB479B8
MANAGER -> BLUE: ..++..++..
MANAGER -> BLUE: #.++..++..
MANAGER -> BLUE: .#########
MANAGER -> BLUE: ##########
MANAGER -> BLUE: ##########
MANAGER -> BLUE: ##########
BLUE -> MANAGER: 0 3 DOWN


=== TURN 1: RED MOVES ===
MANAGER -> RED: 0 3 DOWN 1 OK
MANAGER -> RED: ##########
MANAGER -> RED: ##########
MANAGER -> RED: ##########
MANAGER -> RED: .#########
MANAGER -> RED: #.........
MANAGER -> RED: 9.++..++..
MANAGER -> RED: .67B669999
MANAGER -> RED: 6724898974
MANAGER -> RED: BB31555583
MANAGER -> RED: FB8sB479B8
RED -> MANAGER: 0 5 UP


=== TURN 1: BLUE MOVES ===
MANAGER -> BLUE: 0 5 UP 1 DIES 8
MANAGER -> BLUE: 967B669999
MANAGER -> BLUE: 6724898974
MANAGER -> BLUE: BB31555583
MANAGER -> BLUE: FB8sB479B8
MANAGER -> BLUE: 8.++..++..
MANAGER -> BLUE: ..++..++..
MANAGER -> BLUE: .#########
MANAGER -> BLUE: ##########
MANAGER -> BLUE: ##########
MANAGER -> BLUE: ##########
BLUE -> MANAGER: 1 3 DOWN


=== MATCH END ===
MANAGER -> RED: QUIT
MANAGER -> BLUE: QUIT
```