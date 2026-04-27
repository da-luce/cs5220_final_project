# stratego/CMakeLists.txt
add_library(stratego_core 
    board.cpp 
    engine.cpp 
    state.cpp 
    setup.cpp
)

target_include_directories(stratego_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stratego_core PUBLIC nlohmann_json::nlohmann_json)