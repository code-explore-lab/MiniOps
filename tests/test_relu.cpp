// tests/test_relu.cpp
#include "miniops/operators.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  // ========== 测试用例1：正常样本：负数、负小数、0、正数 ==========
  {
    const float input[] = {-3.0f, -0.5f, 0.0f, 1.0f, 2.5f};
    const int size = sizeof(input) / sizeof(float);
    float output[size];

    miniops::relu(input, output, size);

    const float expect[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.5f};
    for (int i = 0; i < size; i++) {
      assert(fabs(output[i] - expect[i]) < 1e-6f);
    }
    std::cout << "[PASS] normal case: negative / zero / positive\n";
  }

  // ========== 测试用例2：空输入 size=0 ==========
  {
    const float *input = nullptr;
    float *output = nullptr;
    miniops::relu(input, output, 0);
    std::cout << "[PASS] empty input (size=0)\n";
  }

  // ========== 测试用例3：size=1 单元素边界 ==========
  {
    float in1 = -10.f, out1;
    miniops::relu(&in1, &out1, 1);
    assert(fabs(out1 - 0.f) < 1e-6f);

    float in2 = 10.f, out2;
    miniops::relu(&in2, &out2, 1);
    assert(fabs(out2 - 10.f) < 1e-6f);
    std::cout << "[PASS] single element boundary\n";
  }

  std::cout << "\n✅ All ReLU tests passed!\n";
  return 0;
}
