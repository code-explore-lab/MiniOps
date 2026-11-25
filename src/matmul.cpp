// src/matmul.cpp
#include "miniops/operators.hpp"
#include <cstring>

namespace miniops {
void matmul(const float *a, const float *b, float *c, int m, int n, int k) {
  // ========== 边界检查 ==========
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  // 空指针检查：只要需要读取就不能空
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }

  // 朴素 ijk 基准实现，下标和你原来完全一致
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
} // namespace miniops
