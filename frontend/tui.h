#pragma once
#include "../stratego/stratego.h"

namespace tui {

// Initializes ncurses, sets up color pairs and standard terminal config
void init_ncurses();

// Displays the start menu and initializes the board based on user choice.
// Returns true if a game was started, false if the user chose to quit.
bool show_start_menu(stratego::Board& board);

} // namespace tui