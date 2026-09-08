#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t N) {
  if (N <= 1) return;
  double b0 = b[0];
  double *ws = (double*)malloc(2 * N * sizeof(double));
  if (!ws) return;
  double *g = ws;
  double *h = ws + N;

  #pragma omp target data map(to: c[0:N]) map(to: d[0:N]) map(to: e[0:N]) \
                          map(tofrom: a[0:N]) map(from: b[0:N]) \
                          map(alloc: g[0:N]) map(alloc: h[0:N])
  {
    /* g[i] = c[i]*(d[i]+e[i]) for i>=1, g[0]=0 */
    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 0; i < N; i++) g[i] = (i >= 1) ? c[i] * (d[i] + e[i]) : 0.0;

    /* two-buffer inclusive scan (Hillis-Steele); result in A */
    int cur = 0;  /* 0: A=g, 1: A=h */
    for (int64_t s = 1; s < N; s <<= 1) {
      if (cur == 0) {
        #pragma omp target teams distribute parallel for simd
        for (int64_t i = 0; i < N; i++) h[i] = g[i] + (i >= s ? g[i - s] : 0.0);
      } else {
        #pragma omp target teams distribute parallel for simd
        for (int64_t i = 0; i < N; i++) g[i] = h[i] + (i >= s ? h[i - s] : 0.0);
      }
      cur ^= 1;
    }
    if (cur == 0) {
      #pragma omp target teams distribute parallel for simd
      for (int64_t i = 0; i < N; i++) { b[i] = b0 + g[i]; if (i >= 1) a[i] = b0 + g[i - 1] + c[i] * d[i]; }
    } else {
      #pragma omp target teams distribute parallel for simd
      for (int64_t i = 0; i < N; i++) { b[i] = b0 + h[i]; if (i >= 1) a[i] = b0 + h[i - 1] + c[i] * d[i]; }
    }
  }
  free(ws);
}
