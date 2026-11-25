// src/matmul.cpp
#include "miniops/operators.hpp"
#include <algorithm>
#include <cstddef>

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

// 与 matmul_blocking 相同的分块，但把行/列基址提前算好，内层 j 连续访问，
// 避免在热循环里反复计算 (ii+i)*n + (jj+j) 这类下标。
void matmul_blocking_ptr(const float *a, const float *b, float *c, int m, int n,
                         int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[static_cast<std::size_t>(i) * n + j] = 0.0f;
    }
  }

  for (int ii = 0; ii < m; ii += BM) {
    const int ib = std::min(BM, m - ii);
    for (int kk = 0; kk < k; kk += BK) {
      const int kb = std::min(BK, k - kk);
      for (int jj = 0; jj < n; jj += BN) {
        const int jb = std::min(BN, n - jj);
        for (int i = 0; i < ib; ++i) {
          const float *a_row =
              a + static_cast<std::size_t>(ii + i) * k + kk;
          float *c_row = c + static_cast<std::size_t>(ii + i) * n + jj;
          for (int p = 0; p < kb; ++p) {
            const float av = a_row[p];
            const float *b_row =
                b + static_cast<std::size_t>(kk + p) * n + jj;
            for (int j = 0; j < jb; ++j) {
              c_row[j] += av * b_row[j];
            }
          }
        }
      }
    }
  }
}

namespace {

// 计算 C 的一行：c_row[0..n) = a_row[0..k) * B[0..k, 0..n)。
// 在寄存器里累加 JBlock 个输出列，最后只写回 C 一次。
// 内层 t 循环固定长度、访问连续，便于编译器向量化。
template <int JBlock>
void matmul_row_regblock(const float *a_row, const float *b, float *c_row,
                         int n, int k) {
  int j = 0;
  for (; j + JBlock <= n; j += JBlock) {
    float acc[JBlock] = {};
    const float *b_col = b + j;
    for (int p = 0; p < k; ++p) {
      const float av = a_row[p];
      const float *b_row = b_col + static_cast<std::size_t>(p) * n;
      for (int t = 0; t < JBlock; ++t) {
        acc[t] += av * b_row[t];
      }
    }
    for (int t = 0; t < JBlock; ++t) {
      c_row[j + t] = acc[t];
    }
  }

  // n 不能被 JBlock 整除时的尾巴
  for (; j < n; ++j) {
    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
      sum += a_row[p] * b[static_cast<std::size_t>(p) * n + j];
    }
    c_row[j] = sum;
  }
}

template <int JBlock>
void matmul_regblock_impl(const float *a, const float *b, float *c, int m,
                          int n, int k) {
  for (int i = 0; i < m; ++i) {
    matmul_row_regblock<JBlock>(a + static_cast<std::size_t>(i) * k, b,
                                c + static_cast<std::size_t>(i) * n, n, k);
  }
}

// 一次处理 Rows 个输出行：同一行 B 在一个 i 块内被 Rows 个输出行复用，
// 减少 B 从 L3/内存的重复读取；最内层 j 仍然连续，保持自动向量化。
template <int Rows>
void matmul_rows_impl(const float *a, const float *b, float *c, int m, int n,
                      int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[static_cast<std::size_t>(i) * n + j] = 0.0f;
    }
  }

  int i = 0;
  for (; i + Rows <= m; i += Rows) {
    const float *a_rows[Rows];
    float *c_rows[Rows];
    for (int r = 0; r < Rows; ++r) {
      a_rows[r] = a + static_cast<std::size_t>(i + r) * k;
      c_rows[r] = c + static_cast<std::size_t>(i + r) * n;
    }

    for (int p = 0; p < k; ++p) {
      const float *b_row = b + static_cast<std::size_t>(p) * n;
      for (int r = 0; r < Rows; ++r) {
        const float av = a_rows[r][p];
        float *c_row = c_rows[r];
        for (int j = 0; j < n; ++j) {
          c_row[j] += av * b_row[j];
        }
      }
    }
  }

  // m 不能被 Rows 整除时的尾巴行
  for (; i < m; ++i) {
    const float *a_row = a + static_cast<std::size_t>(i) * k;
    float *c_row = c + static_cast<std::size_t>(i) * n;
    for (int p = 0; p < k; ++p) {
      const float av = a_row[p];
      const float *b_row = b + static_cast<std::size_t>(p) * n;
      for (int j = 0; j < n; ++j) {
        c_row[j] += av * b_row[j];
      }
    }
  }
}

} // namespace

void matmul_reg4(const float *a, const float *b, float *c, int m, int n,
                 int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_regblock_impl<4>(a, b, c, m, n, k);
}

void matmul_reg8(const float *a, const float *b, float *c, int m, int n,
                 int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_regblock_impl<8>(a, b, c, m, n, k);
}

void matmul_reg16(const float *a, const float *b, float *c, int m, int n,
                  int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_regblock_impl<16>(a, b, c, m, n, k);
}

void matmul_rows2(const float *a, const float *b, float *c, int m, int n,
                  int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_rows_impl<2>(a, b, c, m, n, k);
}

void matmul_rows4(const float *a, const float *b, float *c, int m, int n,
                  int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_rows_impl<4>(a, b, c, m, n, k);
}

void matmul_rows8(const float *a, const float *b, float *c, int m, int n,
                  int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;
  matmul_rows_impl<8>(a, b, c, m, n, k);
}

// 按输出行并行：线程 i 只写 C 中自己负责的行，写入集合互不相交，
// 不存在多个线程累加同一个输出元素的情况，因此无需 atomic/reduction。
// 未启用 OpenMP 编译时 pragma 被跳过，退化为串行的 reg8 实现。
void matmul_omp(const float *a, const float *b, float *c, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; ++i) {
    matmul_row_regblock<8>(a + static_cast<std::size_t>(i) * k, b,
                           c + static_cast<std::size_t>(i) * n, n, k);
  }
}

// 与 matmul_omp 相同的行划分，但每个线程执行 ikj 内核：内层 j 连续访问
// B 行和 C 行。每行在循环内清零，无需预先清零整个 C。
void matmul_omp_ikj(const float *a, const float *b, float *c, int m, int n,
                    int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (a == nullptr || b == nullptr || c == nullptr)
    return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; ++i) {
    // OpenMP 把循环体 outline 成独立函数后会丢失 n/k 的范围信息，clang 因此
    // 认为内层循环“不可计数”而拒绝向量化。局部 const 副本恢复这个信息。
    const int nn = n;
    const int kk = k;
    const float *a_row = a + static_cast<std::size_t>(i) * kk;
    float *c_row = c + static_cast<std::size_t>(i) * nn;
    for (int j = 0; j < nn; ++j) {
      c_row[j] = 0.0f;
    }
    for (int p = 0; p < kk; ++p) {
      const float av = a_row[p];
      const float *b_row = b + static_cast<std::size_t>(p) * nn;
      for (int j = 0; j < nn; ++j) {
        c_row[j] += av * b_row[j];
      }
    }
  }
}
} // namespace miniops
