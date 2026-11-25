# MiniOps：C++ 高性能 CPU 算子库实战路线

> **目标岗位：** AI 算子库及高性能计算实习生\
> **项目定位：** 面向学习与实习面试的轻量级 C++ CPU 算子库\
> **建议周期：** 14 天完成可展示的 MVP；之后按本文路线继续深化\
> **主要技术：**
> C++17、CMake、GoogleTest（可选）、OpenMP、SIMD、Python/NumPy\
> **核心原则：**
> 先保证正确，再测量性能，最后针对瓶颈优化。没有基准数据，不声称优化成功。

------------------------------------------------------------------------

## 0. 怎么使用这份文档

这份文档是项目的长期执行清单。每次只推进一个小节，不需要依赖聊天上下文。

每完成一项：

1.  在对应复选框打勾。
2.  保留代码、测试命令和测试结果。
3.  记录遇到的问题、定位过程和结论。
4.  不要为了赶进度跳过正确性测试或性能基线。

**每个阶段的交付标准：** 代码可编译、测试可重复、结果可解释、README
有记录。

### 总体里程碑

  里程碑         交付物                   判断是否完成
  -------------- ------------------------ ----------------------------
  M0 环境        C++17 + CMake 工程       一条命令完成构建
  M1 基础算子    ReLU、Add、ReduceSum     覆盖正常和边界输入
  M2 矩阵乘法    朴素 MatMul              与参考结果一致
  M3 Benchmark   稳定计时与报告           多种矩阵规模有数据
  M4 CPU 优化    循环顺序、分块           有优化前后对比
  M5 并行        OpenMP MatMul            正确性通过，记录线程数影响
  M6 SIMD        向量化尝试               有编译信息或性能证据
  M7 项目整理    README、设计说明、报告   别人能按说明复现

------------------------------------------------------------------------

# 1. 项目目标与边界

## 1.1 项目要解决什么问题

神经网络由许多张量运算组成，例如逐元素运算、矩阵乘法、归约、归一化等。算子库负责实现这些基础计算。

本项目先在 CPU 上实现一组基础算子，再逐步探索：

-   数据布局和连续内存访问；
-   循环顺序对缓存局部性的影响；
-   Cache Blocking（分块）；
-   CPU 多线程并行；
-   SIMD/编译器自动向量化；
-   正确性、数值误差和性能测试。

## 1.2 本项目暂时不做什么

第一版不实现完整深度学习框架，不实现自动求导，不做 GPU/NPU
内核，不追求与 MKL、OpenBLAS 或厂商库竞争。

CUDA 是后续扩展，不是第一周的门槛。先把 CPU
版本的计算、内存访问和测量方法掌握。

## 1.3 最终简历项目描述（完成后再使用）

**MiniOps：基于 C++17 的轻量级 CPU 算子库**

使用 C++17 实现 ReLU、Add、ReduceSum、Softmax
和矩阵乘法等基础算子；针对矩阵乘法探索循环顺序、缓存分块、OpenMP
多线程及 SIMD 向量化，并构建正确性测试与
Benchmark，对不同输入规模下的执行时间和加速比进行分析。

注意：只写实际完成的算子和优化。没有测量到的性能提升不要写成成果。

------------------------------------------------------------------------

# 2. 开发环境与工程初始化（M0）

## 2.1 推荐环境

优先使用 Linux。若当前使用 Windows，可使用 WSL2 Ubuntu；也可以使用原生
Linux。

需要：

-   C++17 编译器：GCC 或 Clang；
-   CMake；
-   Git；
-   Python 3；
-   NumPy（用于生成参考结果）；
-   OpenMP（后续阶段使用）。

Ubuntu/WSL2 示例：

``` bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip
python3 -m pip install numpy
```

检查版本：

``` bash
g++ --version
cmake --version
git --version
python3 --version
```

如果系统 Python 不允许直接 pip 安装，可创建虚拟环境：

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

## 2.2 创建仓库

``` bash
mkdir MiniOps
cd MiniOps
git init
mkdir -p include/miniops src tests benchmarks scripts docs
```

建议目录：

``` text
MiniOps/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/
│   └── miniops/
│       ├── tensor.h
│       └── operators.h
├── src/
│   ├── relu.cpp
│   ├── add.cpp
│   ├── reduce.cpp
│   ├── softmax.cpp
│   └── matmul.cpp
├── tests/
│   ├── test_relu.cpp
│   ├── test_add.cpp
│   ├── test_reduce.cpp
│   ├── test_softmax.cpp
│   └── test_matmul.cpp
├── benchmarks/
│   └── benchmark_matmul.cpp
├── scripts/
│   └── compare_numpy.py
└── docs/
    ├── design.md
    └── benchmark_report.md
```

不要一开始就创建一堆空文件。按阶段添加，保持工程始终可编译。

## 2.3 最小 CMakeLists.txt

第一版先只构建库和一个简单测试程序。后续再接入 GoogleTest。

``` cmake
cmake_minimum_required(VERSION 3.16)
project(MiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)
target_link_libraries(benchmark_matmul PRIVATE miniops)
```

构建：

``` bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

调试构建：

``` bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

性能测试必须使用 Release 构建。Debug
构建适合排查问题，不适合拿来比较性能。

------------------------------------------------------------------------

# 3. 先建立最基本的数据约定

在写算子之前，统一数据表示，否则后面很容易把维度、步长和内存布局搞混。

## 3.1 第一版约定

-   数据类型先只支持 `float`。
-   张量数据使用连续的一维 `std::vector<float>` 或调用方提供的
    `float*`。
-   二维矩阵使用 row-major（行优先）布局。
-   第一版只支持连续内存，不支持任意 stride。
-   函数接收维度时，检查维度是否合法。
-   输出内存由调用方分配，函数不得越界访问。

二维矩阵 `A` 的元素 `A[i][j]` 在线性内存中的偏移：

``` cpp
A[i * cols + j]
```

例如一个 2 行 3 列矩阵：

``` text
1 2 3
4 5 6
```

内存顺序是：

``` text
[1, 2, 3, 4, 5, 6]
```

## 3.2 推荐接口风格

``` cpp
#pragma once

namespace miniops {

void relu(const float* input, float* output, int size);

void add(const float* a, const float* b,
         float* output, int size);

void reduce_sum(const float* input, float* output, int size);

void softmax(const float* input, float* output, int size);

void matmul(const float* a, const float* b, float* c,
            int m, int n, int k);

}  // namespace miniops
```

这里矩阵乘法约定：

-   `A` 形状为 `M × K`；
-   `B` 形状为 `K × N`；
-   `C` 形状为 `M × N`；
-   计算 `C = A × B`。

接口文档必须写清楚维度、内存布局、输入输出是否允许别名，以及错误输入如何处理。

------------------------------------------------------------------------

# 4. 第一阶段：实现基础算子（M1）

建议按 ReLU → Add → ReduceSum → Softmax 的顺序实现。

## 4.1 ReLU

数学定义：

\[ y_i = `\max`{=tex}(0, x_i) \]

实现要求：

-   输入和输出长度均为 `size`；
-   `size == 0` 时不访问内存；
-   测试正数、负数、零；
-   明确是否允许输入输出指向同一块内存。第一版可以支持原地计算，但要写进接口说明。

参考实现：

``` cpp
#include "miniops/operators.h"
#include <algorithm>

namespace miniops {

void relu(const float* input, float* output, int size) {
    if (size <= 0) return;

    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

}  // namespace miniops
```

测试数据建议：

``` text
input  = [-3, -0.5, 0, 1, 2.5]
output = [ 0,  0,   0, 1, 2.5]
```

完成标准：

-   [ ] 正数、负数、零测试通过；
-   [ ] 空输入测试通过；
-   [ ] 输出长度和输入长度一致；
-   [ ] 接口约定写入文档。

## 4.2 Add

数学定义：

\[ y_i = a_i + b_i \]

实现逐元素相加。测试：

-   正负数混合；
-   全零；
-   长度为 1；
-   大数组；
-   若支持原地操作，测试别名场景。

完成标准：

-   [ ] 正确处理不同数值；
-   [ ] 输入输出长度约定明确；
-   [ ] 无越界访问。

## 4.3 ReduceSum

数学定义：

\[ s = `\sum`{=tex}\_{i=0}\^{N-1}x_i \]

第一版先实现单线程版本。后续可以探索分块求和和多线程归约。

注意浮点加法不满足严格结合律：

\[ (a+b)+c `\ne `{=tex}a+(b+c) \]

因此不同归约顺序可能产生微小误差。测试不要要求所有浮点结果逐位相同，应使用容差。

完成标准：

-   [ ] 空输入行为明确；
-   [ ] 单元素输入通过；
-   [ ] 正负数混合通过；
-   [ ] 大数组结果与参考实现误差在容差内。

## 4.4 Softmax

数学定义：

\[ y_i = `\frac{e^{x_i}}{\sum_j e^{x_j}}`{=tex} \]

不要直接照公式计算 `exp(x)`。输入值较大时，指数可能溢出。

使用数值稳定版本：

\[ m = `\max`{=tex}\_i x_i \]

\[ y_i = `\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}`{=tex} \]

实现步骤：

1.  找到输入最大值 `m`；
2.  计算 `exp(input[i] - m)`；
3.  求指数结果之和；
4.  每个指数结果除以总和。

测试：

-   [ ] 普通正负数输入；
-   [ ] 所有元素相同；
-   [ ] 含较大的正数；
-   [ ] 含较大的负数；
-   [ ] 输出没有 NaN/Inf；
-   [ ] 输出元素之和约等于 1。

第一版先实现一维 Softmax。之后再考虑二维张量按行计算的 Softmax。

------------------------------------------------------------------------

# 5. 第二阶段：矩阵乘法（M2，项目重点）

这是整个项目最值得深入的部分。

## 5.1 理解计算过程

给定：

\[ A\_{M`\times `{=tex}K}, `\quad `{=tex}B\_{K`\times `{=tex}N} \]

输出：

\[ C\_{M`\times `{=tex}N} \]

每个输出元素：

\[ C\_{ij} = `\sum`{=tex}*{k=0}\^{K-1} A*{ik}B\_{kj} \]

计算复杂度约为：

\[ O(MNK) \]

## 5.2 朴素实现

先实现最容易验证的版本：

``` cpp
void matmul(const float* a, const float* b, float* c,
            int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
```

这里的 `i、j、p` 分别表示输出矩阵的行、列，以及参与点积的维度。

## 5.3 必须覆盖的正确性测试

至少测试以下维度：

  A        B       C
  -------- ------- --------
  1×1      1×1     1×1
  2×3      3×2     2×2
  3×4      4×5     3×5
  64×64    64×64   64×64
  127×65   65×33   127×33

不要只测试正方形矩阵。非正方形矩阵更容易暴露索引错误。

测试时准备一个简单的参考实现，或使用 Python/NumPy 计算参考结果。

## 5.4 边界和接口检查

至少明确处理：

-   `m <= 0`、`n <= 0`、`k <= 0`；
-   空指针；
-   输出缓冲区大小；
-   输入输出内存重叠；
-   维度乘法和内存分配时的整数溢出风险。

如果使用裸指针，函数本身无法知道缓冲区实际长度。因此应在接口文档中写明调用方必须提供足够空间。更完整的工程版本可以使用
`std::span`（C++20）或封装 Tensor 类型；本项目保持 C++17
时可使用指针加长度参数。

------------------------------------------------------------------------

# 6. 第三阶段：Benchmark 基线（M3）

没有基线，就无法证明优化有效。先测朴素 MatMul。

## 6.1 测试要求

-   使用 `Release` 构建；
-   预先分配输入输出内存；
-   计时区间内不要反复分配大块内存；
-   每个规模先预热，再重复多次；
-   报告中记录中位数或平均值，并说明采用哪一种；
-   同一台机器、相同编译器和编译选项下比较；
-   避免机器同时运行大量任务。

建议矩阵规模：

``` text
64 × 64
128 × 128
256 × 256
512 × 512
1024 × 1024
```

机器性能不足时可以减少最大规模。

## 6.2 计时示例

``` cpp
#include <chrono>

using Clock = std::chrono::steady_clock;

auto start = Clock::now();

// 在这里调用算子

auto end = Clock::now();

double ms =
    std::chrono::duration<double, std::milli>(end - start).count();
```

计时器不要只执行一次。每种输入规模重复多次，记录每次耗时，再计算中位数。

## 6.3 记录表格

  版本          矩阵规模   线程数   中位耗时（ms）   相对基线加速比
  ----------- ---------- -------- ---------------- ----------------
  naive              128        1             待测            1.00×
  optimized          128        1             待测           待计算
  naive              512        1             待测            1.00×
  optimized          512        1             待测           待计算

加速比：

\[ Speedup = `\frac{T_{baseline}}{T_{optimized}}`{=tex} \]

所有数据必须来自真实运行，不要为了让报告好看而填造数字。

## 6.4 可选：计算 GFLOP/s

矩阵乘法的浮点运算量近似为：

\[ 2MNK \]

若耗时为秒：

\[ GFLOP/s = `\frac{2MNK}{T \times 10^9}`{=tex} \]

这个指标用于粗略衡量计算吞吐量。不同实现、硬件和计时条件之间比较时，要说明环境。

------------------------------------------------------------------------

# 7. 第四阶段：逐步优化 MatMul（M4）

每次只改一个因素。先提出假设，再实现，再测量。

## 7.1 优化 A：调整循环顺序

朴素版本是 `i-j-p`。由于 B 是行优先存储，内层 `p` 循环访问：

``` cpp
b[p * n + j]
```

相邻迭代访问的地址相隔 `n` 个 float，空间局部性较差。

可以尝试 `i-p-j`：

``` cpp
void matmul_ikj(const float* a, const float* b, float* c,
                int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const float a_ip = a[i * k + p];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}
```

注意：这个版本需要在计算前将 C 初始化为零：

``` cpp
std::fill(c, c + m * n, 0.0f);
```

对比时，初始化成本是否计入计时必须保持一致并在报告中说明。

思考：

-   哪个内层循环更连续地访问内存？
-   C 的数据被重复读写多少次？
-   不同矩阵规模下结果是否相同？
-   小矩阵是否一定更快？

## 7.2 优化 B：Cache Blocking

核心思想：把大矩阵切成小块，让当前参与计算的数据尽可能在缓存中重复使用，减少慢速内存访问。

概念示意：

``` text
A 的一个 block  ×  B 的一个 block
             ↓
       累加到 C 的 block
```

三重分块循环的结构：

``` cpp
for (int ii = 0; ii < m; ii += BM) {
    for (int kk = 0; kk < k; kk += BK) {
        for (int jj = 0; jj < n; jj += BN) {
            // 在当前小块范围内计算
        }
    }
}
```

需要使用 `min` 处理矩阵维度不能整除 block size 的情况。

实验变量：

-   BM、BN、BK 可以先尝试 16、32、64；
-   不要一次同时改变所有变量；
-   不同 CPU 的缓存层级和容量不同，不存在对所有机器都最优的固定块大小。

完成标准：

-   [ ] 非整除维度正确；
-   [ ] 与朴素版本结果在容差范围内；
-   [ ] 测试多个 block size；
-   [ ] 报告记录不同规模的性能变化；
-   [ ] 解释为什么某些配置变慢。

## 7.3 优化 C：避免不必要的内存流量

观察内层循环中是否重复读取相同数据，是否反复写回输出，是否发生不必要的数据复制。

可以尝试：

-   将重复使用的标量加载到局部变量；
-   让内层循环连续访问内存；
-   合理组织输出块的累加；
-   避免在计算热路径中动态分配内存。

每项改动都应单独 benchmark。

------------------------------------------------------------------------

# 8. 第五阶段：OpenMP 多线程（M5）

## 8.1 为什么并行

矩阵乘法中，不同行的输出通常可以并行计算。可以先对输出矩阵的行维度进行并行划分。

## 8.2 CMake 启用 OpenMP

``` cmake
find_package(OpenMP REQUIRED)
target_link_libraries(miniops PUBLIC OpenMP::OpenMP_CXX)
```

如果当前编译器或环境不支持 OpenMP，先检查编译器和运行环境，不要把"加了
pragma"当成已经完成并行优化。

## 8.3 简单并行思路

``` cpp
#pragma omp parallel for
for (int i = 0; i < m; ++i) {
    // 每个线程负责不同的输出行
}
```

注意：

-   不同线程写不同的 C 行，避免数据竞争；
-   不要让多个线程同时累加同一个输出元素；
-   线程数量并非越多越好；
-   小矩阵可能因为线程调度开销而变慢。

## 8.4 实验

尝试线程数：

``` text
1、2、4、8
```

具体数量受 CPU 核心数和环境限制。记录：

-   线程数；
-   矩阵规模；
-   执行时间；
-   加速比；
-   CPU 型号和逻辑核心数。

完成标准：

-   [ ] 单线程与多线程结果一致（容差内）；
-   [ ] ThreadSanitizer 或其他方式辅助排查数据竞争（环境允许时）；
-   [ ] 比较不同线程数；
-   [ ] 分析小矩阵并行变慢的原因。

------------------------------------------------------------------------

# 9. 第六阶段：SIMD 与向量化（M6）

SIMD（Single Instruction, Multiple Data）允许一条指令处理多个数据元素。

学习顺序建议：

1.  先理解连续内存访问；
2.  查看编译器是否自动向量化；
3.  学习基本向量化报告；
4.  再考虑使用编译器 intrinsic 手动实现。

GCC 可以尝试：

``` bash
g++ -O3 -march=native ...
```

查看向量化信息可使用 GCC 的优化报告选项，例如：

``` bash
-fopt-info-vec-optimized
-fopt-info-vec-missed
```

具体报告内容取决于编译器版本和代码结构。

注意：

-   `-march=native` 会针对当前机器生成指令，不保证二进制能在其他 CPU
    上运行；
-   不要在没有验证的情况下开启 `-ffast-math`，它可能改变浮点语义；
-   SIMD 优化后仍需跑完整正确性测试；
-   先观察编译器生成的代码或优化报告，再决定是否手写 intrinsic。

完成标准：

-   [ ] 理解 SIMD 与普通标量循环的区别；
-   [ ] 获得至少一份编译器向量化报告；
-   [ ] 对比启用和未启用优化选项的性能；
-   [ ] 记录精度与可移植性注意事项。

------------------------------------------------------------------------

# 10. 测试策略

测试分为三层。

## 10.1 单元测试

每个算子至少包含：

-   普通输入；
-   空输入或最小输入；
-   边界值；
-   非整齐维度；
-   大输入；
-   错误参数行为。

## 10.2 参考结果对比

可以用 Python/NumPy 生成参考数据。C++ 输出结果后，比较：

\[ \|actual-reference\| `\le `{=tex}atol + rtol
`\cdot `{=tex}\|reference\| \]

其中：

-   `atol`：绝对误差容限；
-   `rtol`：相对误差容限。

对浮点计算，不要无条件使用 `==` 比较。

## 10.3 Sanitizer

调试阶段可使用 AddressSanitizer 和 UndefinedBehaviorSanitizer：

``` bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"

cmake --build build-asan -j
```

运行测试程序，检查越界、释放后使用和未定义行为。

Sanitizer 构建不用于性能比较。

------------------------------------------------------------------------

# 11. 14 天执行安排

这是建议节奏，不是必须在 14 天内硬冲完。理解和验证比打卡重要。

  日期     任务                       当天必须留下的产物
  -------- -------------------------- ----------------------
  Day 1    配好 C++17、CMake、Git     空工程可构建
  Day 2    理解连续内存、行优先布局   `design.md` 初稿
  Day 3    实现 ReLU、Add             代码与测试
  Day 4    实现 ReduceSum、Softmax    代码与测试
  Day 5    理解矩阵乘法索引           手算一个小矩阵例子
  Day 6    实现朴素 MatMul            代码与非正方形测试
  Day 7    完成 NumPy 参考对比        正确性结果
  Day 8    实现 Benchmark             原始基线数据
  Day 9    尝试循环顺序优化           对比表
  Day 10   实现 Cache Blocking        分块版本与测试
  Day 11   测试不同 block size        实验记录
  Day 12   加入 OpenMP                多线程正确性结果
  Day 13   查看 SIMD/向量化报告       编译器报告与对比
  Day 14   整理 README 和报告         可复现的 GitHub 项目

如果只有少量时间，优先完成 Day
1～8，再投递并继续做优化。不要等所有内容都做完才投简历。

------------------------------------------------------------------------

# 12. 每日工作记录模板

每天完成后复制一份：

``` markdown
## 日期：

### 今天完成
-

### 修改的文件
-

### 遇到的问题
-

### 我如何定位
-

### 最终原因
-

### 测试命令与结果
-

### 性能数据（如果有）
-

### 今天真正理解的一个概念
-

### 明天要完成
-
```

这份记录以后可以直接转化成面试中的项目讲解素材。

------------------------------------------------------------------------

# 13. 面试准备：必须能解释的问题

## C++ 与内存

1.  指针和引用有什么区别？
2.  `const float*` 表示什么？
3.  `std::vector` 的内存是否连续？
4.  栈和堆有什么区别？
5.  什么是内存对齐？
6.  什么是未定义行为？

## 矩阵计算

1.  行优先和列优先布局有什么区别？
2.  矩阵乘法的时间复杂度是什么？
3.  为什么 `i-k-j` 循环顺序可能更快？
4.  什么是 Cache Locality？
5.  什么是 Cache Blocking？
6.  为什么矩阵维度不整除 block size 时需要处理边界？

## 并行计算

1.  什么是数据竞争？
2.  什么任务适合并行？
3.  OpenMP 的 `parallel for` 做了什么？
4.  为什么线程越多不一定越快？
5.  什么是线程调度开销？
6.  如何判断一个并行版本是否正确？

## 性能分析

1.  如何设计公平的 Benchmark？
2.  为什么使用 Release 构建？
3.  为什么要预热和多次测量？
4.  加速比如何计算？
5.  为什么小规模输入可能看不到优化收益？
6.  如何区分计算瓶颈和内存访问瓶颈？

## AI 算子基础

1.  什么是 Tensor？
2.  什么是算子？
3.  ReLU、Softmax、MatMul 分别做什么？
4.  Softmax 为什么要减去最大值？
5.  GPU 与 CPU 在并行执行方面有什么差异？
6.  算子正确性和算子性能为什么都重要？

------------------------------------------------------------------------

# 14. 投递策略

## 14.1 不要等项目全部完成

可以采用两条线并行：

-   **投递线：** 先整理 C++ 基础、已有项目和课程经历，尽快投递；
-   **项目线：** 持续推进 MiniOps，每完成一个里程碑就更新仓库。

如果招聘流程很快，14 天项目计划不应成为延迟投递的理由。

## 14.2 简历表述必须真实

项目尚未完成时，可以写：

> 正在使用 C++17 开发轻量级 CPU
> 算子库，已实现/正在实现基础逐元素算子与矩阵乘法，并搭建正确性验证和性能测试框架。

完成优化后，再补充具体技术点和真实数据。

不要写"精通 CUDA""熟悉 GPU 内核优化"，除非确实做过并能回答细节。

## 14.3 实习条件确认

投递前确认：

-   是否仍符合"在读"身份要求；
-   每周能否到岗至少 3 天；
-   能否连续实习至少 3 个月；
-   毕业时间是否与岗位要求冲突。

这些条件可能比多写一个算子更早决定简历能否进入流程。

------------------------------------------------------------------------

# 15. 后续扩展路线（MVP 完成之后）

按岗位相关度逐步扩展：

1.  二维 Softmax（按行计算）；
2.  LayerNorm；
3.  Batch MatMul；
4.  更系统的 Benchmark；
5.  CPU 缓存与硬件计数器分析（如 `perf`，环境支持时）；
6.  与 OpenBLAS 等成熟库比较；
7.  学习 CUDA 基础：线程层级、Grid/Block、Global/Shared Memory；
8.  将 MatMul 或逐元素算子移植到 CUDA；
9.  了解深度学习框架的算子调用与测试方式。

CUDA 阶段需要独立的 NVIDIA GPU、驱动和 CUDA Toolkit
环境。没有合适硬件时，可以先学习概念和阅读示例，但不要伪称完成了 GPU
性能测试。

------------------------------------------------------------------------

# 16. 项目完成验收清单

## 工程

-   [ ] CMake 可以从干净目录构建；
-   [ ] Release 和 Debug 构建均可用；
-   [ ] 目录结构清晰；
-   [ ] 没有提交 build 目录或临时文件；
-   [ ] README 有完整构建和运行命令。

## 算子

-   [ ] ReLU
-   [ ] Add
-   [ ] ReduceSum
-   [ ] Softmax
-   [ ] 朴素 MatMul
-   [ ] 优化 MatMul

## 优化

-   [ ] 循环顺序对比
-   [ ] Cache Blocking
-   [ ] OpenMP
-   [ ] SIMD/自动向量化分析

## 质量

-   [ ] 单元测试通过；
-   [ ] NumPy 参考验证通过；
-   [ ] 非正方形矩阵通过；
-   [ ] 边界维度通过；
-   [ ] Sanitizer 检查（环境允许时）；
-   [ ] Benchmark 可重复运行；
-   [ ] 性能报告包含硬件、编译器、编译选项和原始数据。

## 面试表达

-   [ ] 能画出矩阵乘法的数据布局；
-   [ ] 能解释朴素版本的访存问题；
-   [ ] 能解释分块优化的动机；
-   [ ] 能解释 OpenMP 的并行粒度；
-   [ ] 能说明性能测试方法；
-   [ ] 能诚实说明当前实现的局限。

------------------------------------------------------------------------

# 17. 开始今天的第一项任务

今天只做下面这些，不要同时开十个坑：

1.  安装并检查 C++ 编译器、CMake、Git、Python；
2.  创建 `MiniOps` 仓库；
3.  按第 2 节创建最小 CMake 工程；
4.  写一个最小 `main.cpp` 或 benchmark 程序，确认能链接 `miniops`；
5.  执行构建命令并保存输出；
6.  提交第一次 Git commit。

建议提交信息：

``` bash
git add .
git commit -m "chore: initialize MiniOps project"
```

**第一天的完成标准只有一个：一个干净的 C++17 + CMake 工程能成功构建。**

不要第一天就冲
CUDA、SIMD、矩阵乘法和深度学习框架。先把地基打好，工程不会因为你同时打开八个教程就自动长出性能。
