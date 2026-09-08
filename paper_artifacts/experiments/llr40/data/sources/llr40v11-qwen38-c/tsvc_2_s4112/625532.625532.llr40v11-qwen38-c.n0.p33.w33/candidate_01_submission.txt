#include <stdint.h>
void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  const int64_t nmain = n - 256;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < nmain; ++i) {
    __builtin_prefetch(&b[ip[i+64]], 0, 1);
    __builtin_prefetch(&b[ip[i+128]], 0, 1);
    a[i] += b[ip[i]] * 2.0;
  }
  for (int64_t i = nmain; i < n; ++i) a[i] += b[ip[i]] * 2.0;
}
