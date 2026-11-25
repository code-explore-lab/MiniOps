#pragma once
namespace miniops {
void relu(const float *input, float *output, int size);
void add(const float *a, const float *b, float *output, int size);
void reduce_sum(const float *input, float *output, int size);
void softmax(const float *input, float *output, int size);
void matmul(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_ikj(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_blocking(const float *a, const float *b, float *c, int m, int n,
                     int k);
void matmul_blocking_ptr(const float *a, const float *b, float *c, int m, int n,
                         int k);
void matmul_reg4(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_reg8(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_reg16(const float *a, const float *b, float *c, int m, int n,
                  int k);
void matmul_rows2(const float *a, const float *b, float *c, int m, int n,
                  int k);
void matmul_rows4(const float *a, const float *b, float *c, int m, int n,
                  int k);
void matmul_rows8(const float *a, const float *b, float *c, int m, int n,
                  int k);
// 按输出行并行；未启用 OpenMP 编译时退化为串行的 reg8 实现。
void matmul_omp(const float *a, const float *b, float *c, int m, int n, int k);
// 同样按输出行并行，但每个线程使用 ikj 内核（内层 j 连续、可自动向量化）。
void matmul_omp_ikj(const float *a, const float *b, float *c, int m, int n,
                    int k);
} // namespace miniops