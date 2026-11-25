// tests/test_add.cpp
#include "miniops/operators.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>


int main() {
  const float eps = 1e-6f;

  // Case1: 正负混合
  {
    const float a[] = {1.0f, -2.5f, 0.0f, 3.3f};
    const float b[] = {-1.0f, 0.5f, 5.0f, -1.3f};
    const float expect[] = {0.f, -2.f, 5.f, 2.f};
    int size = sizeof(a) / sizeof(float);
    float out[size];
    miniops::add(a, b, out, size);
    for (int i = 0; i < size; i++) {
      assert(fabs(out[i] - expect[i]) < eps);
    }
    std::cout << "[PASS] mixed positive & negative\n";
  }

  // Case2: 全零
  {
    const float a[] = {0, 0, 0, 0};
    const float b[] = {0, 0, 0, 0};
    const float expect[] = {0, 0, 0, 0};
    int size = sizeof(a) / sizeof(float);
    float out[size];
    miniops::add(a, b, out, size);
    for (int i = 0; i < size; i++) {
      assert(fabs(out[i] - expect[i]) < eps);
    }
    std::cout << "[PASS] all zero\n";
  }

  // Case3: size = 1
  {
    float a = 5.2f, b = -3.2f, out;
    miniops::add(&a, &b, &out, 1);
    assert(fabs(out - 2.0f) < eps);
    std::cout << "[PASS] size=1 single element\n";
  }

  // Case4: 大数组
  {
    const int N = 256;
    float a[N], b[N], out[N];
    float expect[N];
    for (int i = 0; i < N; i++) {
      a[i] = i * 0.1f;
      b[i] = -i * 0.2f;
      expect[i] = a[i] + b[i];
    }
    miniops::add(a, b, out, N);
    for (int i = 0; i < N; i++) {
      assert(fabs(out[i] - expect[i]) < eps);
    }
    std::cout << "[PASS] large array (256 elements)\n";
  }

  // Case5: 原地别名 in-place：output == a
  {
    float buf[] = {1, 2, 3, 4};
    const float b[] = {10, 20, 30, 40};
    const float expect[] = {11, 22, 33, 44};
    int size = sizeof(buf) / sizeof(float);
    // output 和 a 是同一个指针，原地写
    miniops::add(buf, b, buf, size);
    for (int i = 0; i < size; i++) {
      assert(fabs(buf[i] - expect[i]) < eps);
    }
    std::cout << "[PASS] inplace alias (output == a)\n";
  }

  std::cout << "\n✅ All add tests passed!\n";
  return 0;
}
