find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIRS})

# Build the ASCII interactive frontend
add_executable(play_ascii play_ascii.cpp)
target_link_libraries(play_ascii PRIVATE stratego_core ${CURSES_LIBRARIES})
