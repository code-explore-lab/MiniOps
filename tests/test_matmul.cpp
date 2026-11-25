// tests/test_matmul.cpp
//
// 所有 MatMul 实现都必须与朴素参考实现一致。
// 这里用显式检查而不是 assert：Release 构建定义了 NDEBUG，assert 会失效。

#include "miniops/operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr float kRelEps = 1e-4f;

using MatmulFn = void (*)(const float *, const float *, float *, int, int, int);

struct Candidate {
  const char *name;
  MatmulFn fn;
};

const Candidate kCandidates[] = {
    {"naive", &miniops::matmul},
    {"ikj", &miniops::matmul_ikj},
    {"blocking", &miniops::matmul_blocking},
    {"blocking_ptr", &miniops::matmul_blocking_ptr},
    {"reg4", &miniops::matmul_reg4},
    {"reg8", &miniops::matmul_reg8},
    {"reg16", &miniops::matmul_reg16},
    {"rows2", &miniops::matmul_rows2},
    {"rows4", &miniops::matmul_rows4},
    {"rows8", &miniops::matmul_rows8},
    {"omp", &miniops::matmul_omp},
    {"omp_ikj", &miniops::matmul_omp_ikj},
};
constexpr int kCandidateCount =
    static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

// 辅助：朴素参考实现
void matmul_ref(const float *a, const float *b, float *c, int m, int n, int k) {
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

// 相对 + 绝对容差比较。不同累加顺序会带来少量浮点差异。
bool allclose(const float *lhs, const float *rhs, int len) {
  for (int i = 0; i < len; i++) {
    const float scale =
        std::max({1.0f, std::fabs(lhs[i]), std::fabs(rhs[i])});
    if (std::fabs(lhs[i] - rhs[i]) > kRelEps * scale) {
      return false;
    }
  }
  return true;
}

std::vector<float> make_matrix(int rows, int cols, int mod, float scale) {
  std::vector<float> v(static_cast<std::size_t>(rows) * cols);
  for (std::size_t i = 0; i < v.size(); i++) {
    v[i] = static_cast<float>(static_cast<int>(i % mod)) * scale;
  }
  return v;
}

void check_case(const char *label, int m, int n, int k,
                const std::vector<float> &a, const std::vector<float> &b,
                bool verbose) {
  std::vector<float> ref(static_cast<std::size_t>(m) * n);
  matmul_ref(a.data(), b.data(), ref.data(), m, n, k);

  for (const Candidate &cand : kCandidates) {
    // 用哨兵值预填充，确保实现必须自己写满整个输出。
    std::vector<float> out(static_cast<std::size_t>(m) * n, -12345.0f);
    cand.fn(a.data(), b.data(), out.data(), m, n, k);
    if (!allclose(out.data(), ref.data(), m * n)) {
      std::cerr << "[FAIL] " << label << " (" << cand.name << ")\n";
      std::exit(1);
    }
  }
  if (verbose) {
    std::cout << "[PASS] " << label << " (" << kCandidateCount << " impls)\n";
  }
}

void check_invalid_inputs() {
  std::vector<float> buf(64, 1.0f);
  for (const Candidate &cand : kCandidates) {
    cand.fn(buf.data(), buf.data(), buf.data(), 0, 2, 3);
    cand.fn(buf.data(), buf.data(), buf.data(), 2, -1, 3);
    cand.fn(buf.data(), buf.data(), buf.data(), 2, 2, 0);
    cand.fn(nullptr, buf.data(), buf.data(), 2, 2, 2);
    cand.fn(buf.data(), nullptr, buf.data(), 2, 2, 2);
    cand.fn(buf.data(), buf.data(), nullptr, 2, 2, 2);
  }
  std::cout << "[PASS] dim <= 0 / nullptr safe (" << kCandidateCount
            << " impls)\n";
}

void run_shape_cases(bool verbose) {
  // Case1: 1x1 @ 1x1 = 1x1（行数少于线程数）
  check_case("1x1 @ 1x1", 1, 1, 1, {2.5f}, {4.0f}, verbose);

  // Case2: 2x3 @ 3x2 = 2x2
  check_case("2x3 @ 3x2", 2, 2, 3, {1, 2, 3, 4, 5, 6},
             {10, 20, 30, 40, 50, 60}, verbose);

  // Case3: 3x4 @ 4x5 = 3x5
  check_case("3x4 @ 4x5", 3, 5, 4, make_matrix(3, 4, 20, 0.1f),
             make_matrix(4, 5, 20, 0.2f), verbose);

  // Case4: 64x64 @ 64x64
  check_case("64x64 @ 64x64", 64, 64, 64, make_matrix(64, 64, 10, 1.0f),
             make_matrix(64, 64, 7, 1.0f), verbose);

  // Case5: 127x65 @ 65x33，非对称且三维都不能被分块大小整除
  check_case("127x65 @ 65x33", 127, 33, 65, make_matrix(127, 65, 13, 0.01f),
             make_matrix(65, 33, 17, 0.02f), verbose);

  // Case6: 37x29 @ 29x41，再覆盖一组小质数尺寸
  check_case("37x29 @ 29x41", 37, 41, 29, make_matrix(37, 29, 11, 0.05f),
             make_matrix(29, 41, 19, 0.03f), verbose);
}

} // namespace

int main() {
  run_shape_cases(true);
  check_invalid_inputs();

#ifdef _OPENMP
  // 同一套形状在多个线程数下重复验证：并行只改变行划分，
  // 每个输出元素仍由单线程按相同顺序累加，结果应与串行一致。
  const int thread_counts[] = {1, 2, 3, 4, 8, 16};
  for (int threads : thread_counts) {
    omp_set_num_threads(threads);
    run_shape_cases(false);
    std::cout << "[PASS] all shapes with " << threads << " OpenMP thread(s)\n";
  }
#else
  std::cout << "[INFO] built without OpenMP; matmul_omp runs serially\n";
#endif

  std::cout << "\n✅ All matmul tests passed (" << kCandidateCount
            << " implementations)!\n";
  return 0;
}
