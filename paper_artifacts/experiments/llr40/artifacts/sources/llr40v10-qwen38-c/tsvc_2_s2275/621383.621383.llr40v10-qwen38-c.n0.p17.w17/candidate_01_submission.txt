/* TSVC tsvc_2 s2275: aa (N x N, row-major) += bb*cc elementwise; a[i] = b[i] + c[i]*d[i].
 * The nested (i,j) loop touches each idx = j*N+i exactly once, so it is a plain
 * elementwise FMA over N*N independent elements. Flatten it (unit stride, vectorizes
 * and parallelizes), then do the length-N vector FMA. */
#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                       const double *restrict bb, const double *restrict c,
                       const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t total = N * N;
  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < total; ++idx) {
    aa[idx] = aa[idx] + bb[idx] * cc[idx];
  }
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) {
    a[i] = b[i] + c[i] * d[i];
  }
}
