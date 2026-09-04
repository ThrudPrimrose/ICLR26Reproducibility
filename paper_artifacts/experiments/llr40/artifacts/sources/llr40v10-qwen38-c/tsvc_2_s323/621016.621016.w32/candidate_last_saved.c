#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  printf("PROBE LEN_1D=%lld max_threads=%d\n", (long long)LEN_1D, omp_get_max_threads());
  fflush(stdout);
  double t0 = omp_get_wtime();
  for (int64_t i = 1; i < LEN_1D; ++i) {
    a[i] = b[i - 1] + c[i] * d[i];
    b[i] = a[i] + c[i] * e[i];
  }
  double t1 = omp_get_wtime();
  printf("PROBE ref_seq_ns=%lld b0=%f a1=%f b_last=%f\n",
         (long long)((t1 - t0) * 1e9), b[0], a[1], b[LEN_1D-1]);
  fflush(stdout);
}
