#include <stdint.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;
  const int64_t N2 = N * N;

  #pragma omp target teams distribute map(to: b[0:N], c[0:N]) map(tofrom: a[0:N], aa[0:N2]) map(to: bb[0:N2])
  for (int64_t i = 0; i < N; ++i) {
    const double ain = a[i] + b[i] * c[i];
    a[i] = ain;
    double prev = aa[i];          /* aa[0,i] */
    for (int64_t j = 1; j < N; ++j) {
      prev += bb[j * N + i] * ain;
      aa[j * N + i] = prev;
    }
  }
}
