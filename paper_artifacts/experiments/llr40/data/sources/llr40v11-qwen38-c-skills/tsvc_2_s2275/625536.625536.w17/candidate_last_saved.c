#include <stdint.h>
#include <omp.h>

/*
 * TSVC tsvc_2 s2275 (fp64):
 *   for i in 0..N: for j in 0..N: aa[j*N+i] += bb[j*N+i]*cc[j*N+i];  a[i] = b[i] + c[i]*d[i];
 *
 * Dependence vectors: none. The aa update touches every index j*N+i exactly once
 * (a complete elementwise FMA over the whole 2D array), so the (i,j) nest is a
 * permutation of one flat unit-stride loop and the a[i] update is independent of
 * it. Rewrite as two flat passes, each fully parallel + vectorizable.
 */
void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
  const int64_t NN = LEN_2D * LEN_2D;
  #pragma omp parallel for simd schedule(static)
  for (int64_t k = 0; k < NN; ++k) {
    aa[k] = aa[k] + bb[k] * cc[k];
  }
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_2D; ++i) {
    a[i] = b[i] + c[i] * d[i];
  }
}
