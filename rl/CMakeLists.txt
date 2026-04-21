find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create a library for the RL/PPO Agent
add_library(stratego_rl stratego_tiny_agent.cpp)
target_include_directories(stratego_rl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stratego_rl PUBLIC stratego_core "${TORCH_LIBRARIES}")

add_executable(train train.cpp)

# Link Torch, the RL agent, and the core Stratego library
target_link_libraries(train PRIVATE stratego_rl stratego_core "${TORCH_LIBRARIES}")