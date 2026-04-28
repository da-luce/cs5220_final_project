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

# Janky Perlmutter fix
find_package(CUDAToolkit QUIET)
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
    find_library(_nvtx_lib nvToolsExt
        HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64 ENV CUDA_HOME PATH_SUFFIXES lib64 lib)
    if(_nvtx_lib)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_nvtx_lib}")
    endif()
endif()

# Stub for Intel JIT profiling symbols required by libtorch_cpu.so on Perlmutter
add_library(ittnotify_stub SHARED ittnotify_stub.cpp)

# --- Subdirectories ---
add_subdirectory(stratego)
add_subdirectory(rl)
add_subdirectory(ui)

# --- The Unified Test Binary ---
enable_testing()

add_executable(unified_test
    # Feature Extraction implementation + test
    rl/encoding/board.cpp
    rl/encoding/actions.cpp
    rl/encoding/test_board.cpp
    rl/encoding/test_actions.cpp
    
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