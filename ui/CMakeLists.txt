# This CMake file builds the terminal-based game executable.

# Define the 'play' executable target.
# Note: play_ascii.cpp includes other .cpp files directly (e.g., orchestrator.cpp).
# This is an unconventional practice. A better long-term solution would be to
# compile those files into their own libraries and include their headers.
add_executable(play
    play_ascii.cpp
    stratego_ui.cpp
    tui.cpp # Assumes a tui.cpp exists to implement tui.h
)

# The 'play' executable needs to find headers from the rl and runtime directories.
target_include_directories(play PRIVATE ${CMAKE_SOURCE_DIR})

# Link the executable against the core game logic library (stratego_core),
# the RL library (rl_core), LibTorch, and the ncurses library.
target_link_libraries(play PRIVATE 
    stratego_core 
    rl_core
    "${TORCH_LIBRARIES}"
    ${CURSES_LIBRARIES}
)