#include <stdint.h>
#include <stddef.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
  const size_t N = (size_t)LEN_2D;
  const size_t NN = N * N;

  #pragma omp target teams distribute map(to: bb[0:NN], cc[0:NN]) map(tofrom: aa[0:NN])
  for (size_t idx = 0; idx < NN; ++idx) {
    aa[idx] = aa[idx] + bb[idx] * cc[idx];
  }

  #pragma omp target teams distribute map(to: b[0:N], c[0:N], d[0:N]) map(from: a[0:N])
  for (size_t i = 0; i < N; ++i) {
    a[i] = b[i] + c[i] * d[i];
  }
}
