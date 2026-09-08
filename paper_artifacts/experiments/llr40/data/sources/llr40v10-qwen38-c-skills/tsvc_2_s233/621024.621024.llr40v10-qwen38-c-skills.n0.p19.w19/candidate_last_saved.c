#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;

  /* Phase 1: column scans. aa[j,i] = aa[j-1,i] + cc[j,i], j = 8..N-1.
   * Dependence crosses j (within a column); i is free. */
#pragma omp parallel for schedule(static)
  for (int64_t i = 8; i < N; ++i) {
    double acc = aa[7 * N + i];
    for (int64_t j = 8; j < N; ++j) {
      acc += cc[j * N + i];
      aa[j * N + i] = acc;
    }
  }

  /* Phase 2: row scans. bb[j,i] = bb[j,i-1] + cc[j,i], i = 8..N-1.
   * Dependence crosses i (within a row); j is free. */
#pragma omp parallel for schedule(static)
  for (int64_t j = 8; j < N; ++j) {
    double *brow = bb + j * N;
    const double *crow = cc + j * N;
    double acc = brow[7];
    for (int64_t i = 8; i < N; ++i) {
      acc += crow[i];
      brow[i] = acc;
    }
  }
}
