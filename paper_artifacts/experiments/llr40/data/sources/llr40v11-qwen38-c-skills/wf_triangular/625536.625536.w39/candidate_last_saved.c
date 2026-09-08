#include <stdint.h>
#include <omp.h>
#include <stdio.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  double wt0 = omp_get_wtime();
  const int64_t n = LEN_2D;
  const int64_t B = 128;
  const int64_t R = (n + B - 1) / B;
  #pragma omp parallel
  for (int64_t m = 0; m < 2 * R - 1; m++) {
    int64_t tr_lo = m - (R - 1); if (tr_lo < 0) tr_lo = 0;
    int64_t tr_hi = m;           if (tr_hi > R - 1) tr_hi = R - 1;
    #pragma omp for schedule(static, 1)
    for (int64_t tr = tr_lo; tr <= tr_hi; tr++) {
      const int64_t tc = m - tr;
      int64_t i0 = tr * B; if (i0 < 1) i0 = 1;   /* row 0 is never updated */
      const int64_t i1 = (tr + 1) * B < n ? (tr + 1) * B : n;
      const int64_t j0 = tc * B;
      const int64_t j1 = (tc + 1) * B < n ? (tc + 1) * B : n;
      if (j1 - 1 < i0) continue;                 /* no domain cell in this tile */
      for (int64_t i = i0; i < i1; i++) {
        int64_t j_lo = j0; if (j_lo < i) j_lo = i;
        double *row = a + i * n;
        const double *north = row - n;
        double west = row[j_lo - 1];
        for (int64_t j = j_lo; j < j1; j++) {
          west = row[j] + north[j] + west;
          row[j] = west;
        }
      }
    }
  }
  double wt1 = omp_get_wtime();
  fprintf(stdout, "KERNEL_TIME_MS=%.3f threads=%d\n", (wt1-wt0)*1e3, omp_get_max_threads());
  fflush(stdout);
}
