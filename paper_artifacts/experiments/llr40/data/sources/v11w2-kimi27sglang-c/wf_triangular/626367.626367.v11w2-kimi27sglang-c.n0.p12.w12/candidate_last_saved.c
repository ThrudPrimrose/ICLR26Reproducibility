#include <stdint.h>
#include <omp.h>

#ifndef WF_TILE
#define WF_TILE 192
#endif

static inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }

static void tile(double *restrict a, int64_t N, int64_t T,
                 int64_t ti, int64_t tj) {
  int64_t r0 = max64(ti * T, 1);
  int64_t c0 = tj * T;
  int64_t r1 = min64((ti + 1) * T, N);
  int64_t c1 = min64((tj + 1) * T, N);
  if (r0 >= r1 || c0 >= c1) return;
  for (int64_t i = r0; i < r1; ++i) {
    int64_t jstart = max64(i, c0);
    if (jstart >= c1) continue;
    double carry;
    if (jstart == i) {
      carry = a[i * N + (i - 1)];
    } else {
      carry = a[i * N + (jstart - 1)];
    }
    for (int64_t j = jstart; j < c1; ++j) {
      carry += a[(i - 1) * N + j] + a[i * N + j];
      a[i * N + j] = carry;
    }
  }
}

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;

  if (N <= 2048) {
    for (int64_t i = 1; i < N; ++i) {
      double *restrict row = a + i * N;
      const double *restrict prow = a + (i - 1) * N;
      double carry = row[i - 1];
      for (int64_t j = i; j < N; ++j) {
        carry += prow[j] + row[j];
        row[j] = carry;
      }
    }
    return;
  }

  const int64_t T = WF_TILE;
  const int64_t nt = (N + T - 1) / T;

  for (int64_t s = 0; s <= 2 * (nt - 1); ++s) {
    int64_t ti0 = max64(0, s - (nt - 1));
    int64_t ti1 = min64(s, nt - 1);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t ti = ti0; ti <= ti1; ++ti) {
      int64_t tj = s - ti;
      if (tj < ti || tj >= nt) continue;
      tile(a, N, T, ti, tj);
    }
  }
}
