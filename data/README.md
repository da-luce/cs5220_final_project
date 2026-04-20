# Stratego Deployment Marginal Probabilities

### Overview
This dataset contains the marginal probability distributions of Stratego piece placements within the 4x10 deployment zone, derived from expert gameplay analysis.

- [Source](https://stratego.fandom.com/wiki/Advanced:_Stratego_Setup_Analysis#:~:text=You%20can%20look%20at%20the,things%20I%20noticed%2C%20for%20example%3A)
- [Sheet](https://docs.google.com/spreadsheets/d/1IVLos-My-QCDB18wBmvPn7S88xJ2wOcTL3imsSXLPG8/edit?pli=1&gid=471730407#gid=471730407)


### Data Structure
- **Top Level Keys**: Piece name (e.g., `"MARSHAL"`, `"FLAG"`, `"BOMB"`).
- **Row Indices**: `"0"` (Back row) through `"3"` (Front row).
- **Column Arrays**: A 10-element list representing columns 0-9 (Left to Right).
- **Values**: Percentages (0.00 - 100.00).

### Key Constraints
- **Grid Size**: 4 rows × 10 columns.
- **Orientation**: Indices are relative to the player's perspective.
- **Lakes**: Deployment zone is located in front of the lakes (Rows 4-5 are no-man's land).

### Other Resources

- [Top 20 common game setups at Gravon site](https://web.archive.org/web/20201128164944/http://forum.stratego.com/topic/4470-top-20-common-game-setups-at-gravon-site/)
