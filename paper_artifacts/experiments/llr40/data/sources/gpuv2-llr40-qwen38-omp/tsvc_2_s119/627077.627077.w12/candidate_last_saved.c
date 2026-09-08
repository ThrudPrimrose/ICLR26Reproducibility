#include <stdint.h>
#include <stdlib.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 1) return;
  const int64_t total = N * N;

  /* small N: host compute (GPU transfer/launch overhead dominates) */
  if (N < 1024) {
    #pragma omp parallel for schedule(static, 1)
    for (int64_t k = 0; k < 2 * N - 1; ++k) {
      const int64_t d = k - (N - 1);
      int64_t i0 = 1 + d; if (i0 < 1) i0 = 1;
      int64_t i1 = N - 1; const int64_t alt = N - 1 + d; if (alt < i1) i1 = alt;
      double r = aa[(i0 - 1) * N + (i0 - d - 1)];
      for (int64_t i = i0; i <= i1; ++i) {
        const int64_t idx = i * N + (i - d);
        r += bb[idx];
        aa[idx] = r;
      }
    }
    return;
  }

  /* gather the 2N-1 boundary values the recurrence needs (row 0, col 0) */
  double *init = (double *)malloc(16 * (size_t)N);
  for (int64_t j = 0; j < N; ++j) init[j] = aa[j];
  for (int64_t i = 1; i < N; ++i) init[N + i - 1] = aa[i * N];

  #pragma omp target map(to: bb[0:total]) map(to: init[0:2*N-1]) map(from: aa[N:total-N])
  {
    #pragma omp teams distribute static(chunk(1))
    for (int64_t k = 0; k < 2 * N - 1; ++k) {
      const int64_t d = k - (N - 1);
      int64_t i0 = 1 + d; if (i0 < 1) i0 = 1;
      int64_t i1 = N - 1; const int64_t alt = N - 1 + d; if (alt < i1) i1 = alt;
      double r = (d >= 0) ? (d == 0 ? init[0] : init[N + d - 1]) : init[-d];
      for (int64_t i = i0; i <= i1; ++i) {
        const int64_t idx = i * N + (i - d);
        r += bb[idx];
        aa[idx] = r;
      }
    }
  }
  free(init);
}
