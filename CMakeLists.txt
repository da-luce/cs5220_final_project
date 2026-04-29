cmake_minimum_required(VERSION 3.14)
project(StrategoProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build options ---

# Path to the NCCL shared library. Set when the system NCCL is absent or too old and
# you need to point at one bundled elsewhere (e.g. inside a PyTorch wheel).
# Leave empty to skip NCCL linking entirely.
set(NCCL_LIB     "" CACHE FILEPATH "Path to libnccl.so (leave empty to skip)")
set(NCCL_INCLUDE "" CACHE PATH     "Path to NCCL include directory (leave empty to skip)")

# Build the ittnotify no-op stub. Only needed when linking against a libtorch_cpu.so
# that was compiled with Intel VTune support and the VTune runtime is not installed.
# See ittnotify_stub.cpp for details.
option(BUILD_ITT_STUB "Build the ittnotify no-op stub library" OFF)

# --- Dependencies ---
include(FetchContent)

# JSON
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# GTest
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.12.1)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# LibTorch — set CMAKE_PREFIX_PATH to the torch cmake directory before invoking cmake.
# See README for example commands for Perlmutter and local builds.
find_package(Torch REQUIRED)

# OpenMP — Apple Clang doesn't bundle it; point CMake at Homebrew's libomp
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

add_definitions(-DPROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

# If the CUDA toolkit is present but nvToolsExt isn't exported as a CMake target,
# create a minimal INTERFACE target so libtorch can link against it.
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

if(BUILD_ITT_STUB)
    add_library(ittnotify_stub SHARED perlmutter/ittnotify_stub.cpp)
endif()

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
    OpenMP::OpenMP_CXX
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
    $<$<BOOL:${BUILD_ITT_STUB}>:ittnotify_stub>
    $<$<BOOL:${NCCL_LIB}>:${NCCL_LIB}>
)

# Include directories so headers like "rl/encoding/board.h" and "stratego/board.h" are found
target_include_directories(unified_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(unified_test DISCOVERY_MODE PRE_TEST)