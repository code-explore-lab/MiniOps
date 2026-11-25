// src/matmul.cpp
#include "miniops/operators.hpp"
#include <algorithm>

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
// 存储：**行主序，一行元素在内存上连续**
void matmul_ikj(const float *a, const float *b, float *c, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;

  // 必须清零！！
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      //`a[i*k + p]`：i 固定，p 增加 → 连续
      const float a_ip = a[i * k + p]; // 载入寄存器，只读一次
      //`b[p*n + j]`：p 固定，j 增加 → **j 是最内层，b [p][0],b [p][1],b [p][2]…
      // 连续访问**
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += a_ip * b[p * n + j];
      }
    }
  }
}
// 块大小，你可以调：16/32/64
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;

void matmul_blocking(const float *a, const float *b, float *c, int m, int n,
                     int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;

  // 先清零整个C（ikj累加模式必须）
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
    }
  }

  // ========== 外层分块循环：tiling ==========
  for (int ii = 0; ii < m; ii += BM) {
    for (int kk = 0; kk < k; kk += BK) {
      for (int jj = 0; jj < n; jj += BN) {

        // 当前块实际尺寸，处理不能整除的尾巴
        int ib = std::min(BM, m - ii);
        int kb = std::min(BK, k - kk);
        int jb = std::min(BN, n - jj);

        // ========== 内层：小块内 ikj 乘法 ==========
        for (int i = 0; i < ib; ++i) {
          for (int p = 0; p < kb; ++p) {
            float aval = a[(ii + i) * k + (kk + p)];
            for (int j = 0; j < jb; ++j) {
              c[(ii + i) * n + (jj + j)] += aval * b[(kk + p) * n + (jj + j)];
            }
          }
        }
      }
    }
  }
}
} // namespace miniops
