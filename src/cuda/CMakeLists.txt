# src/cuda/CMakeLists.txt
#
# Conditional CUDA build. If nvcc isn't available (e.g. on macOS dev machines),
# this subdirectory is a no-op so the rest of the project still builds normally.

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler not found — skipping GPU targets in src/cuda.")
    return()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Library: device kernels + host-side bridge
add_library(stratego_gpu STATIC
    stratego_gpu.cu
    state_convert.cpp
)
set_target_properties(stratego_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(stratego_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(stratego_gpu PUBLIC stratego_core)

# Correctness-check binary (CPU vs GPU move generation)
add_executable(test_gpu_moves test_gpu.cu)
set_target_properties(test_gpu_moves PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_gpu_moves PRIVATE stratego_gpu stratego_core)
target_include_directories(test_gpu_moves PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
