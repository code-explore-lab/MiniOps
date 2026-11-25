#include "miniops/operators.hpp"
#include <algorithm>
namespace miniops {
void relu(const float *input, float *output, int size) {
  if (size <= 0)
    return;
  for (int i = 0; i < size; i++) {
    output[i] = std::max(0.0f, input[i]);
  }
}
} // namespace miniops