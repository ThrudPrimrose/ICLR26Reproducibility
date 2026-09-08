/* TSVC s233: aa column-wise scan (parallel over i), bb row-wise scan (parallel over j).
 * aa[j][i] = aa[7][i]  + cumsum_{k=8..j} cc[k][i]   (chain over j, per column i)
 * bb[j][i] = bb[j][7]  + cumsum_{k=8..i} cc[j][k]   (chain over i, per row j)
 * Same addition order as the reference -> bit-identical results. */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;

  #pragma omp parallel for schedule(static)
  for (int64_t i = 8; i < N; ++i) {
    double run = aa[7 * N + i];
    for (int64_t j = 8; j < N; ++j) {
      run += cc[j * N + i];
      aa[j * N + i] = run;
    }
  }

  #pragma omp parallel for schedule(static)
  for (int64_t j = 8; j < N; ++j) {
    double run = bb[j * N + 7];
    for (int64_t i = 8; i < N; ++i) {
      run += cc[j * N + i];
      bb[j * N + i] = run;
    }
  }
}
