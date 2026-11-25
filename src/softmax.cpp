#include "miniops/operators.hpp"
#include <cmath>
namespace miniops {
void softmax(const float *input, float *output, int size) {
  if (size <= 0) {
    return;
  }
  float m = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = std::exp(input[i] - m);
    sum += output[i];
  }
  for (int i = 0; i < size; i++) {
    output[i] = output[i] / sum;
  }
}
} // namespace miniops
