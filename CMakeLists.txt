cmake_minimum_required(VERSION 3.16)
project(MiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成 compile_commands.json，供 clangd / C/C++ 插件解析 include 路径
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenMP 依赖编译器自带的运行时。当前默认的 TDM-GCC 10.3 只有 omp.h、
# 没有 libgomp，开启后会在链接阶段失败；所以默认关闭，OpenMP 构建使用
# scripts/build-openmp.bat（clang++ 22 + libomp）。
option(MINIOPS_ENABLE_OPENMP "Build MatMul with OpenMP" OFF)

if(MINIOPS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(MSVC)
    set(MINIOPS_WARNING_FLAGS /W4)
else()
    set(MINIOPS_WARNING_FLAGS -Wall -Wextra -Wpedantic)
endif()

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
    ${MINIOPS_WARNING_FLAGS}
)

if(MINIOPS_ENABLE_OPENMP)
    # PUBLIC：测试和 benchmark 也需要 OpenMP 头文件、编译选项和运行时。
    target_link_libraries(miniops PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled (${OpenMP_CXX_FLAGS})")
else()
    message(STATUS "OpenMP disabled (MINIOPS_ENABLE_OPENMP=OFF)")
endif()
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

add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)
target_link_libraries(benchmark_matmul PRIVATE miniops)
target_compile_options(benchmark_matmul PRIVATE
    ${MINIOPS_WARNING_FLAGS}
)
