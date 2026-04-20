# Create a library for the core Stratego logic
add_library(stratego_core stratego.cpp setup_generator.cpp)
target_include_directories(stratego_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stratego_core PUBLIC nlohmann_json::nlohmann_json)

# Build the test executable
add_executable(test_stratego test_stratego.cpp)
target_link_libraries(test_stratego PRIVATE stratego_core)

# Register individual tests so they can be run and tracked separately via `ctest`
add_test(NAME TestOutOfBounds COMMAND test_stratego out_of_bounds)
add_test(NAME TestImmobilePieces COMMAND test_stratego immobile_pieces)
add_test(NAME TestStandardMoves COMMAND test_stratego standard_moves)
add_test(NAME TestScoutMoves COMMAND test_stratego scout_moves)
add_test(NAME TestCombatResolutions COMMAND test_stratego combat_resolutions)
add_test(NAME TestSaveLoad COMMAND test_stratego save_load)
add_test(NAME TestMoveLimit COMMAND test_stratego move_limit)
add_test(NAME TestTwoSquaresRule COMMAND test_stratego two_squares_rule)
add_test(NAME TestMoreSquaresRule COMMAND test_stratego more_squares_rule)