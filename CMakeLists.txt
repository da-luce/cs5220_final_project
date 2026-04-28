cmake_minimum_required(VERSION 3.14)
project(StrategoProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)

# JSON
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# GTest
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.12.1)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# LibTorch
find_package(Torch REQUIRED)

add_definitions(-DPROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

# --- Subdirectories ---
add_subdirectory(src/stratego)
add_subdirectory(src/rl)
add_subdirectory(src/ui)

# --- The Arena Binary ---
add_executable(arena
    src/runtime/arena.cpp
    src/runtime/policy_ucc.cpp
)

target_link_libraries(arena 
    PRIVATE 
    stratego_core
    rl_core
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

target_include_directories(arena PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- The Unified Test Binary ---
enable_testing()

add_executable(unified_test
    # RL tests
    src/rl/encoding/test_board.cpp
    src/rl/encoding/test_actions.cpp
    
    # Stratego Logic tests
    src/stratego/test_board.cpp
    src/stratego/test_engine.cpp
    src/stratego/test_state.cpp
    src/stratego/test_setup.cpp
)

# Rename 'unified_test' to just './test' in the build directory
set_target_properties(unified_test PROPERTIES OUTPUT_NAME "test")

# Link everything
target_link_libraries(unified_test 
    PRIVATE 
    stratego_core             # Defined in src/stratego/CMakeLists.txt
    rl_core                   # Defined in src/rl/CMakeLists.txt
    GTest::gtest_main 
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

# Include directories so headers like "rl/encoding/board.h" and "stratego/board.h" are found
target_include_directories(unified_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(unified_test)