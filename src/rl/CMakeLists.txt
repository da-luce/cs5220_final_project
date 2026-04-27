cmake_minimum_required(VERSION 3.14)

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
)

# This allows #include "networks/..." to work if rl is in include path
target_include_directories(rl_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Training executable
add_executable(train_stratego main_train.cpp)
target_link_libraries(train_stratego PRIVATE rl_core)
