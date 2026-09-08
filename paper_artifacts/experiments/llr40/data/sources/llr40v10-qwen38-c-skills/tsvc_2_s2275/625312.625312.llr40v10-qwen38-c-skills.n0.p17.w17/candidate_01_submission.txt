#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
  const int64_t N = LEN_2D * LEN_2D;
  /* Every aa[idx] update is independent of every other: the reference's
     (i, j) ordering is just a permutation of the same N elementwise ops.
     Flattened it is a unit-stride FMA chain, trivially vectorizable and
     threadable. a[i] = b[i] + c[i]*d[i] is independent per i as well, and
     independent of the aa work, so it runs as a second flat loop. */
#pragma omp parallel for simd schedule(static)
  for (int64_t k = 0; k < N; ++k) {
    aa[k] = aa[k] + bb[k] * cc[k];
  }
#pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_2D; ++i) {
    a[i] = b[i] + c[i] * d[i];
  }
}
