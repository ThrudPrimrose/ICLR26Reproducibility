#include <stdint.h>
#include <omp.h>

/* TSVC s2233: two column-wise scans.
 * aa[j][i] = aa[j-1][i] + cc[j][i]  (j,i in [8,N): chain along rows j per column i)
 * bb[i][j] = bb[i-1][j] + cc[i][j]  (i,j in [8,N): chain along rows i per column j)
 * Both are: for each column c, v = X[7][c]; for r=8..N-1: v += cc[r][c]; X[r][c] = v.
 * Columns are independent -> parallel over columns. */

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 9) { /* tiny: plain serial (no offload overhead) */
    for (int64_t i = 8; i < N; ++i) {
      for (int64_t j = 8; j < N; ++j) aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
      for (int64_t j = 8; j < N; ++j) bb[i * N + j] = bb[(i - 1) * N + j] + cc[i * N + j];
    }
    return;
  }
  const int64_t M2 = N - 8;      /* number of rows/cols in [8,N) */
  const int64_t n2 = N * N;

  #pragma omp target data \
      map(to: cc[0:n2]) \
      map(to: aa[7 * N + 8 : M2]) \
      map(to: bb[7 * N + 8 : M2]) \
      map(from: aa[8 * N + 8 : M2]) \
      map(from: bb[8 * N + 8 : M2]) \
      map(from: aa[9 * N : n2 - 9 * N]) \
      map(from: bb[9 * N : n2 - 9 * N])
  {
    #pragma omp teams distribute parallel for
    for (int64_t c = 0; c < M2; ++c) {
      double v = aa[7 * N + 8 + c];
      const double *src = cc + (8 * N + 8 + c);
      double *dst = aa + (8 * N + 8 + c);
      for (int64_t r = 0; r < M2; ++r) { v += src[r * N]; dst[r * N] = v; }
    }
    #pragma omp teams distribute parallel for
    for (int64_t c = 0; c < M2; ++c) {
      double v = bb[7 * N + 8 + c];
      const double *src = cc + (8 * N + 8 + c);
      double *dst = bb + (8 * N + 8 + c);
      for (int64_t r = 0; r < M2; ++r) { v += src[r * N]; dst[r * N] = v; }
    }
  }
}
