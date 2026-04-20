cmake_minimum_required(VERSION 3.10)
project(StrategoRL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable testing with CTest
enable_testing()

# Create a library for the core Stratego logic
add_library(stratego_core stratego/stratego.cpp)
target_include_directories(stratego_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/stratego)

# Build the test executable
add_executable(test_stratego stratego/test_stratego.cpp)
target_link_libraries(test_stratego PRIVATE stratego_core)

# Register the test so it can be run via `ctest` or `make test`
add_test(NAME TestStratego COMMAND test_stratego)

# Include the frontend module
add_subdirectory(frontend)