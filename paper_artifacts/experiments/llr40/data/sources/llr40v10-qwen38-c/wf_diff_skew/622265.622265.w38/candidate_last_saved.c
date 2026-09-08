#include <stdint.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  for (int64_t i = 1; i < LEN_2D; ++i) {
    for (int64_t j = 0; j < LEN_2D - 1; ++j) {
      a[i * LEN_2D + j] = a[i * LEN_2D + j] + a[(i - 1) * LEN_2D + j] + a[(i - 1) * LEN_2D + (j + 1)];
    }
  }
}
