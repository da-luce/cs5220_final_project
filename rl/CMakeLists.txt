cmake_minimum_required(VERSION 3.14)

find_package(MPI)

# Apple Clang doesn't bundle OpenMP; point CMake at Homebrew's libomp if present.
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(rl_core
    encoding/board.cpp
    encoding/actions.cpp
    environment/stratego_env.cpp
    environment/batched_stratego_env.cpp
    agent/ppo_agent.cpp
    networks/model.cpp
    networks/torsos/cnn.cpp
    networks/heads/heads.cpp
    policy_neural.cpp
)

target_link_libraries(rl_core
    PRIVATE
    stratego_core
    "${TORCH_LIBRARIES}"
    $<$<BOOL:${MPI_FOUND}>:${MPI_CXX_LIBRARIES}>
    $<$<BOOL:${NCCL_LIB}>:${NCCL_LIB}>
    OpenMP::OpenMP_CXX
)

target_include_directories(rl_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<BOOL:${MPI_FOUND}>:${MPI_CXX_INCLUDE_DIRS}>
    $<$<BOOL:${NCCL_INCLUDE}>:${NCCL_INCLUDE}>
)

add_executable(train_stratego main_train.cpp)
target_link_libraries(train_stratego
    PRIVATE
    rl_core
    $<$<BOOL:${BUILD_ITT_STUB}>:ittnotify_stub>
    $<$<BOOL:${MPI_FOUND}>:${MPI_CXX_LIBRARIES}>
    $<$<BOOL:${NCCL_LIB}>:${NCCL_LIB}>
)

if(NCCL_LIB)
    target_compile_definitions(train_stratego PRIVATE USE_NCCL)
endif()
