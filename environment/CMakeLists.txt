# Build the test executable for the environment
add_executable(test_stratego_env test_stratego_env.cpp)
target_link_libraries(test_stratego_env PRIVATE stratego_core)

# Register individual tests so they can be run and tracked separately via `ctest`
add_test(NAME EnvTestChannelCount COMMAND test_stratego_env channel_count)
add_test(NAME EnvTestBasicPiecePlacement COMMAND test_stratego_env basic_piece_placement)
add_test(NAME EnvTestUnrevealedAndEmptyChannels COMMAND test_stratego_env unrevealed_and_empty_channels)
add_test(NAME EnvTestPerspectiveFlip COMMAND test_stratego_env perspective_flip)
add_test(NAME EnvTestActionEncoding COMMAND test_stratego_env action_encoding)