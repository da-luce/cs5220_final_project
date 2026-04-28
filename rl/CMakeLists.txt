cmake_minimum_required(VERSION 3.14)

find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# NCCL (on Perlmutter: module load nccl)
find_library(NCCL_LIB nccl HINTS $ENV{NCCL_HOME}/lib /usr/local/cuda/lib64)
find_path(NCCL_INCLUDE nccl.h HINTS $ENV{NCCL_HOME}/include /usr/local/cuda/include)

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
target_compile_definitions(train_stratego PRIVATE USE_NCCL)
