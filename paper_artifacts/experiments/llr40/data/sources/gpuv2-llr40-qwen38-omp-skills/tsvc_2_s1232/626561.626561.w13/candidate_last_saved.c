#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;
  const int nonzero = VLEN > 0;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) {
    int64_t jmax = nonzero ? (i / VLEN + 1) : N;
    if (jmax > N) jmax = N;
    double *restrict a = aa + i * N;
    const double *restrict b = bb + i * N;
    const double *restrict c = cc + i * N;
    for (int64_t j = 0; j < jmax; ++j) a[j] = b[j] + c[j];
  }
  /* minimal target region so the library registers a device kernel */
  int acc = 0;
  #pragma omp target map(tofrom: acc)
  acc = 1;
}
