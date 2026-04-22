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

# ncurses
find_package(Curses REQUIRED)

# --- Subdirectories ---
add_subdirectory(stratego)
# (frontend subdirectory likely contains its own executable, so we leave it)
add_subdirectory(frontend)

# --- The Unified Test Binary ---
enable_testing()

add_executable(unified_test
    # Feature Extraction implementation + test
    features/features.cpp
    features/test_features.cpp
    
    # Stratego Logic tests (files located in stratego/ folder)
    stratego/test_board.cpp
    stratego/test_engine.cpp
    stratego/test_state.cpp
    stratego/test_setup.cpp
)

# Rename 'unified_test' to just './test' in the build directory
set_target_properties(unified_test PROPERTIES OUTPUT_NAME "test")

# Link everything
target_link_libraries(unified_test 
    PRIVATE 
    stratego_core             # Defined in stratego/CMakeLists.txt
    GTest::gtest_main 
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

# Include directories so headers like "features.h" and "board.h" are found
target_include_directories(unified_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/features
)

include(GoogleTest)
gtest_discover_tests(unified_test)