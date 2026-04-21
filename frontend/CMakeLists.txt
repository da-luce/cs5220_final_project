find_package(Curses REQUIRED)
find_package(Torch REQUIRED)

# Build the ASCII interactive frontend
add_executable(play play_ascii.cpp tui.cpp stratego_ui.cpp)
target_include_directories(play PRIVATE ${CURSES_INCLUDE_DIR})
target_link_libraries(play PRIVATE stratego_core ${CURSES_LIBRARIES} "${TORCH_LIBRARIES}")
set_target_properties(play PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
