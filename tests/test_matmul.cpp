// tests/test_matmul.cpp
#include "miniops/operators.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>


static constexpr float eps = 1e-4f;

// 辅助：朴素参考实现（和算子逻辑一致，用于校验）
static void matmul_ref(const float *a, const float *b, float *c, int m, int n,
                       int k) {
  if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c)
    return;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float s = 0.f;
      for (int p = 0; p < k; ++p) {
        s += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = s;
    }
  }
}

// 辅助：逐元素对比
static bool allclose(const float *lhs, const float *rhs, int len) {
  for (int i = 0; i < len; i++) {
    if (std::fabs(lhs[i] - rhs[i]) > eps) {
      return false;
    }
  }
  return true;
}

int main() {
  // Case1: 1x1 @ 1x1 = 1x1
  {
    const float a[] = {2.5f};
    const float b[] = {4.0f};
    float c[1], ref[1];
    miniops::matmul(a, b, c, 1, 1, 1);
    matmul_ref(a, b, ref, 1, 1, 1);
    assert(allclose(c, ref, 1));
    std::cout << "[PASS] 1x1 @ 1x1\n";
  }

  // Case2: 2x3 @ 3x2 = 2x2
  {
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {10, 20, 30, 40, 50, 60};
    float c[4], ref[4];
    miniops::matmul(a, b, c, 2, 2, 3);
    matmul_ref(a, b, ref, 2, 2, 3);
    assert(allclose(c, ref, 4));
    std::cout << "[PASS] 2x3 @ 3x2\n";
  }

  // Case3: 3x4 @ 4x5 = 3x5
  {
    const int m = 3, n = 5, k = 4;
    float a[m * k], b[k * n], c[m * n], ref[m * n];
    // 填充简单序列
    for (int i = 0; i < m * k; i++)
      a[i] = float(i + 1) * 0.1f;
    for (int i = 0; i < k * n; i++)
      b[i] = float(i + 1) * 0.2f;
    miniops::matmul(a, b, c, m, n, k);
    matmul_ref(a, b, ref, m, n, k);
    assert(allclose(c, ref, m * n));
    std::cout << "[PASS] 3x4 @ 4x5\n";
  }

  // Case4: 64x64 @ 64x64 = 64x64
  {
    const int m = 64, n = 64, k = 64;
    float a[m * k], b[k * n], c[m * n], ref[m * n];
    for (int i = 0; i < m * k; i++)
      a[i] = float(i % 10);
    for (int i = 0; i < k * n; i++)
      b[i] = float((i + 3) % 7);
    miniops::matmul(a, b, c, m, n, k);
    matmul_ref(a, b, ref, m, n, k);
    assert(allclose(c, ref, m * n));
    std::cout << "[PASS] 64x64 @ 64x64\n";
  }

  // Case5: 127x65 @ 65x33 = 127x33（非对称，重点！）
  {
    const int m = 127, n = 33, k = 65;
    float a[m * k], b[k * n], c[m * n], ref[m * n];
    for (int i = 0; i < m * k; i++)
      a[i] = float(i % 13) * 0.01f;
    for (int i = 0; i < k * n; i++)
      b[i] = float(i % 17) * 0.02f;
    miniops::matmul(a, b, c, m, n, k);
    matmul_ref(a, b, ref, m, n, k);
    assert(allclose(c, ref, m * n));
    std::cout << "[PASS] 127x65 @ 65x33\n";
  }

  // ========== 边界 & 接口检查 ==========
  // m/n/k <= 0，不崩溃
  {
    float dummy;
    const float arr[] = {1, 2, 3};
    miniops::matmul(arr, arr, &dummy, 0, 2, 3);
    miniops::matmul(arr, arr, &dummy, 2, -1, 3);
    miniops::matmul(arr, arr, &dummy, 2, 2, 0);
    std::cout << "[PASS] dim <=0 safe\n";
  }
  // 空指针输入，不崩溃
  {
    float c[4];
    miniops::matmul(nullptr, c, c, 2, 2, 2);
    miniops::matmul(c, nullptr, c, 2, 2, 2);
    miniops::matmul(c, c, nullptr, 2, 2, 2);
    std::cout << "[PASS] nullptr safe\n";
  }

  std::cout << "\n✅ All matmul tests passed!\n";
  return 0;
}
