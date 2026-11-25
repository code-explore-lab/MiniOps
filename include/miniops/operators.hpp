#pragma once
namespace miniops {
void relu(const float *input, float *output, int size);
void add(const float *a, const float *b, float *output, int size);
void reduce_sum(const float *input, float *output, int size);
void softmax(const float *input, float *output, int size);
void matmul(const float *a, const float *b, float *c, int m, int n, int k);
} // namespace miniops