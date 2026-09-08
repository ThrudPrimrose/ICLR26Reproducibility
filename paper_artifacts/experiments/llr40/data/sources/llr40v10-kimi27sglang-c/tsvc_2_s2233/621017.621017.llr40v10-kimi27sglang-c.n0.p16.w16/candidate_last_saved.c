#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {

  // First loop: column-wise recurrence.  Columns are independent, so parallelize over i.
  #pragma omp parallel for
  for (int64_t i = 8; i < LEN_2D; ++i) {
    for (int64_t j = 8; j < LEN_2D; ++j) {
      aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
    }
  }

  // Second loop: row-wise recurrence.  Rows are dependent, so sequential i; vectorize j.
  for (int64_t i = 8; i < LEN_2D; ++i) {
    #pragma omp simd
    for (int64_t j = 8; j < LEN_2D; ++j) {
      bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
    }
  }
}
