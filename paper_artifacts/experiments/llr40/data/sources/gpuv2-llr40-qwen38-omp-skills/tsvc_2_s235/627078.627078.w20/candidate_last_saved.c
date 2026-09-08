#include <stdint.h>
#include <omp.h>
#include <stdio.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  double t0 = omp_get_wtime();
  #pragma omp target data \
      map(tofrom: a[0:LEN_2D]) \
      map(to: b[0:LEN_2D]) \
      map(to: c[0:LEN_2D]) \
      map(tofrom: aa[0:LEN_2D * LEN_2D]) \
      map(to: bb[0:LEN_2D * LEN_2D])
  {
    double t1 = omp_get_wtime();
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < LEN_2D; ++i) {
      a[i] += b[i] * c[i];
      for (int64_t j = 1; j < LEN_2D; ++j) {
        aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * a[i];
      }
    }
    double t2 = omp_get_wtime();
    printf("PROBE L=%ld in_ms=%.3f compute_ms=%.3f out_ms=%.3f ondev=%d\n", (long)LEN_2D,
           (t1 - t0) * 1e3, (t2 - t1) * 1e3, (omp_get_wtime() - t2) * 1e3, 1);
    fflush(stdout);
  }
}
