cmake_minimum_required(VERSION 3.14)
project(StrategoProject CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch JSON Library (v3.11.3 fixes the policy error)
include(FetchContent)
FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# 2. Fetch GoogleTest
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.12.1)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 3. Find ncurses for the frontend
find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIRS})

# 4. Enable CTest project-wide
enable_testing()

# 5. Orchestrate the subdirectories
add_subdirectory(stratego)
add_subdirectory(frontend)
# add_subdirectory(environment)