cmake_minimum_required(VERSION 3.16)
project(MiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成 compile_commands.json，供 clangd / C/C++ 插件解析 include 路径
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(miniops
    src/relu.cpp
    src/add.cpp
    src/reduce.cpp 
    src/softmax.cpp
    src/matmul.cpp
)

target_include_directories(miniops
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(miniops PRIVATE
    -Wall -Wextra -Wpedantic
)
add_executable(test_relu
    tests/test_relu.cpp
)
target_link_libraries(test_relu PRIVATE miniops)
add_executable(test_add
    tests/test_add.cpp
)
target_link_libraries(test_add PRIVATE miniops)
add_executable(test_reduce
    tests/test_reduce.cpp
)
target_link_libraries(test_reduce PRIVATE miniops)
add_executable(test_softmax
    tests/test_softmax.cpp
)
target_link_libraries(test_softmax PRIVATE miniops)
add_executable(test_matmul
    tests/test_matmul.cpp
)
target_link_libraries(test_matmul PRIVATE miniops)
