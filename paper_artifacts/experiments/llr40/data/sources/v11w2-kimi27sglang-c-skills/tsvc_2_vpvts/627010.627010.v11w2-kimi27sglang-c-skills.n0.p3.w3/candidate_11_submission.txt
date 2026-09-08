#include <stdint.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  if (__builtin_expect(LEN_1D < 4096, 1)) {
    #pragma omp simd
    for (int64_t i = 0; i < LEN_1D; ++i) {
      a[i] += b[i] * S;
    }
    return;
  }
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] += b[i] * S;
  }
}
