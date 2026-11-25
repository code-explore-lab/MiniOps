#include "miniops/operators.hpp"
#include <cassert>
#include <cmath>
#include <iostream>


int main() {
  const float eps = 1e-5f;

  // 辅助：判断是否是非法浮点数
  auto is_valid = [](float v) {
    return std::isfinite(v); // 不是NaN、不是Inf
  };

  // Case1：普通正负数混合输入
  {
    const float x[] = {1.0f, -2.0f, 0.5f, 3.0f};
    const int size = sizeof(x) / sizeof(float);
    float out[size];
    miniops::softmax(x, out, size);

    float sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (!is_valid(out[i]))
        ok = false;
      sum += out[i];
    }
    assert(ok);
    assert(fabs(sum - 1.0f) < eps);
    std::cout << "[PASS] mixed positive & negative\n";
  }

  // Case2：所有元素完全相同
  {
    const float x[] = {5.0f, 5.0f, 5.0f};
    const int size = sizeof(x) / sizeof(float);
    float out[size];
    miniops::softmax(x, out, size);

    float sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (!is_valid(out[i]))
        ok = false;
      sum += out[i];
      assert(fabs(out[i] - 1.0f / size) < eps);
    }
    assert(ok);
    assert(fabs(sum - 1.0f) < eps);
    std::cout << "[PASS] all elements equal\n";
  }

  // Case3：包含很大正数（测试数值稳定，不溢出Inf）
  {
    const float x[] = {990.0f, 991.0f, 992.0f};
    const int size = sizeof(x) / sizeof(float);
    float out[size];
    miniops::softmax(x, out, size);

    float sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (!is_valid(out[i]))
        ok = false;
      sum += out[i];
    }
    assert(ok);
    assert(fabs(sum - 1.0f) < eps);
    std::cout << "[PASS] large positive values (no Inf)\n";
  }

  // Case4：包含很大负数（不会出现NaN）
  {
    const float x[] = {-1000.0f, -999.0f, -1001.0f};
    const int size = sizeof(x) / sizeof(float);
    float out[size];
    miniops::softmax(x, out, size);

    float sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (!is_valid(out[i]))
        ok = false;
      sum += out[i];
    }
    assert(ok);
    assert(fabs(sum - 1.0f) < eps);
    std::cout << "[PASS] large negative values (no NaN)\n";
  }

  // Case5：空输入兜底
  {
    float dummy;
    miniops::softmax(nullptr, &dummy, 0);
    std::cout << "[PASS] empty input size=0\n";
  }

  std::cout << "\n✅ All softmax tests passed!\n";
  return 0;
}
