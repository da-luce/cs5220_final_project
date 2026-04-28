cmake_minimum_required(VERSION 3.14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# NCCL — use the versioned .so bundled with PyTorch; system NCCL is too old
set(NCCL_LIB "/global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2")
set(NCCL_INCLUDE "/global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/nvidia/nccl/include")

# Define RL library
add_library(rl_core
    encoding/board.cpp
    encoding/actions.cpp
    environment/stratego_env.cpp
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
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIB}
    OpenMP::OpenMP_CXX
)

# This allows #include "networks/..." to work if rl is in include path
target_include_directories(rl_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${NCCL_INCLUDE}
)

# Training executable
add_executable(train_stratego main_train.cpp)
target_link_libraries(train_stratego PRIVATE rl_core ittnotify_stub ${MPI_CXX_LIBRARIES} ${NCCL_LIB})
if(NCCL_LIB AND NCCL_INCLUDE)
    target_compile_definitions(train_stratego PRIVATE USE_NCCL)
endif()
