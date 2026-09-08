/* plain C port for local timing comparison (not submitted) */
#include <stdint.h>
void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  double sum = 0.0;
  for (int64_t i = 0; i < LEN_1D; ++i) { sum += a[i]; b[i] = sum; }
}
