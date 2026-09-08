#include <stdint.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  #pragma omp target teams distribute parallel for map(tofrom: a[0:LEN_1D], b[0:LEN_1D], ip[0:LEN_1D])
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] += b[ip[i]] * 2.0;
  }
}
