# Create a library for the core Stratego logic
add_library(stratego_core 
    board.cpp 
    engine.cpp 
    state.cpp 
    setup.cpp
)

target_include_directories(stratego_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stratego_core PUBLIC nlohmann_json::nlohmann_json)

# Build the combined GoogleTest executable
add_executable(test_stratego 
    test_board.cpp
    test_engine.cpp 
    test_game.cpp 
    test_setup.cpp
)

# Link the core library AND the GoogleTest main runner
target_link_libraries(test_stratego PRIVATE stratego_core GTest::gtest_main)

# Turn on CMake's native GoogleTest integration
include(GoogleTest)
gtest_discover_tests(test_stratego)