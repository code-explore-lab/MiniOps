#include "miniops/operators.hpp"
namespace miniops {
void reduce_sum(const float *input, float *output, int size) {
  *output = 0.0f;
  if (size <= 0) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    sum += input[i];
  }
  *output = sum;
}
} // namespace miniops