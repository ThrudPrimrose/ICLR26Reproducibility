#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void readbw(const double *restrict x, const double *restrict y, const double *restrict z,
                   const double *restrict w, const double *restrict v, int64_t n, int nt, double *out_ms, double *out_bw) {
  double sink = 0.0;
  double t0 = omp_get_wtime();
#pragma omp parallel for simd num_threads(nt) schedule(static) reduction(+:sink)
  for (int64_t i = 0; i < n; ++i) sink += x[i] + y[i] + z[i] + w[i] + v[i];
  double t1 = omp_get_wtime();
  if (sink > 1e30) printf("x\n");
  *out_ms = 1e3 * (t1 - t0);
  *out_bw = 40.0 * n / 1e9 / (t1 - t0);
}

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  if (LEN_1D > 1000000) {
    int nts[12] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96};
    for (int k = 0; k < 12; ++k) {
      double ms, bw;
      readbw(c, d, e, a, b, LEN_1D, nts[k], &ms, &bw);
      printf("rd nt=%2d ms=%8.3f bw=%6.1f GB/s\n", nts[k], ms, bw);
    }
  }
  double t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    b[i] = d[i] * e[i];
    a[i] += b[i] * c[i];
  }
  double t1 = omp_get_wtime();
  printf("kernel24 ms=%.3f bw=%.1f GB/s\n", 1e3 * (t1 - t0), 56.0 * LEN_1D / 1e9 / (t1 - t0));
  fflush(stdout);
}
