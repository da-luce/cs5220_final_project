# This CMake file builds the terminal-based game executable.

include(FetchContent)

# FTXUI
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(ftxui)

# Define the 'play' executable target.
# Note: play_ascii.cpp includes other .cpp files directly (e.g., orchestrator.cpp).
# This is an unconventional practice. A better long-term solution would be to
# compile those files into their own libraries and include their headers.
add_executable(play
    play_ascii.cpp
    stratego_ui.cpp
)

# The 'play' executable needs to find headers from the rl and runtime directories.
target_include_directories(play PRIVATE ${CMAKE_SOURCE_DIR})

# Link the executable against the core game logic library (stratego_core),
# the RL library (rl_core), LibTorch, and FTXUI.
target_link_libraries(play PRIVATE
    stratego_core
    rl_core
    "${TORCH_LIBRARIES}"
    $<$<BOOL:${BUILD_ITT_STUB}>:ittnotify_stub>
    ftxui::screen
    ftxui::dom
    ftxui::component
)