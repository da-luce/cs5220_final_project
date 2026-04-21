cmake_minimum_required(VERSION 3.11)
project(StrategoRL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Dependencies via FetchContent ---
# This automatically downloads required libraries at configure time.
include(FetchContent)

# nlohmann/json for parsing setup data
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3 # Use a specific version for reproducibility
)
FetchContent_MakeAvailable(nlohmann_json)

# Enable testing with CTest
enable_testing()

# Include the core Stratego logic and tests
add_subdirectory(stratego)

# Include the frontend module
add_subdirectory(frontend)

# Include the RL module
add_subdirectory(rl)
