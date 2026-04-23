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

# The 'play' executable needs to find headers from the root 'agents' directory.
# The 'stratego' headers are found by linking against the 'stratego_core' library.
target_include_directories(play PRIVATE ${CMAKE_SOURCE_DIR}/agents)

# Link the executable against the core game logic library (stratego_core)
# and the ncurses library (which is found in the root CMakeLists.txt).
target_link_libraries(play PRIVATE stratego_core ${CURSES_LIBRARIES})