#include "miniops/operators.hpp"
#include <cassert>
#include <cmath>
#include <iostream>


int main() {
  const float eps = 1e-5f;
  float out;

  // ✅ 1. 空输入 size=0
  {
    miniops::reduce_sum(nullptr, &out, 0);
    assert(fabs(out - 0.0f) < eps);
    std::cout << "[PASS] empty input (size=0)\n";
  }

  // ✅ 2. 单元素输入
  {
    const float x[] = {7.2f};
    miniops::reduce_sum(x, &out, 1);
    assert(fabs(out - 7.2f) < eps);
    std::cout << "[PASS] single element\n";
  }

  // ✅ 3. 正负数混合
  {
    const float x[] = {2.5f, -1.5f, 4.0f, -3.0f};
    // 2.5 -1.5 +4 -3 = 2.0
    miniops::reduce_sum(x, &out, 4);
    assert(fabs(out - 2.0f) < eps);
    std::cout << "[PASS] mixed positive and negative\n";
  }

  // ✅ 4. 大数组：和参考实现对比，误差在容差内
  {
    const int N = 2048;
    float x[N];
    float ref_sum = 0.0f;
    // 参考实现：串行累加作为真值
    for (int i = 0; i < N; i++) {
      x[i] = (i - 1024) * 0.001f;
      ref_sum += x[i];
    }
    miniops::reduce_sum(x, &out, N);
    assert(fabs(out - ref_sum) < eps);
    std::cout << "[PASS] large array (2048 elements), within tolerance\n";
  }

  std::cout << "\n✅ All reduce_sum tests passed!\n";
  return 0;
}
