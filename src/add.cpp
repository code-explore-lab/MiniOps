// src/add.cpp
#include "miniops/operators.hpp"
#include <algorithm>

namespace miniops {
void add(const float *a, const float *b, float *output, int size) {
  if (size <= 0) {
    return;
  }
  for (int i = 0; i < size; ++i) {
    output[i] = a[i] + b[i];
  }
}
} // namespace miniops
