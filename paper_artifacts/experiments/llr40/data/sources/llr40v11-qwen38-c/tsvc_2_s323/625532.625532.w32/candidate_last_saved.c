#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

/* s323: a[i] = b[i-1] + c[i]*d[i]; b[i] = a[i] + c[i]*e[i]  (i = 1..N-1)
 * => b is a prefix sum: b[i] = b[0] + sum_{k=1..i} (c[k]*d[k] + c[k]*e[k]).
 *
 * Block-parallel scan:
 *   Phase 1: each thread scans its contiguous block serially from 0, storing
 *            LOCAL values into a,b (chain = 2 FMAs/element, same rounding as
 *            the C oracle which compiles a[i]=FMA(c[i],d[i],b[i-1]) etc.).
 *   Combine: exclusive prefix over per-thread block sums, + b[0].
 *   Phase 2: vectorized in-place pass adds each element's offset. */

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n < 2) return;

  int nt = omp_get_max_threads();
  if (nt < 1) nt = 1;

  int64_t S = (n - 1 + nt - 1) / nt;
  if (S < 64) S = 64;
  const int64_t nblk = (n - 1 + S - 1) / S;

  double *sums = (double *)malloc((size_t)nblk * sizeof(double));

  #pragma omp parallel num_threads(nt)
  {
    const int t = omp_get_thread_num();
    const int64_t start = 1 + (int64_t)t * S;
    int64_t i = start;
    const int64_t end = start + S < n ? start + S : n;
    double s = 0.0;

    for (; i + 3 < end; i += 4) {
      double a0 = fma(c[i],     d[i],     s);
      double s0 = fma(c[i],     e[i],     a0);
      double a1 = fma(c[i + 1], d[i + 1], s0);
      double s1 = fma(c[i + 1], e[i + 1], a1);
      double a2 = fma(c[i + 2], d[i + 2], s1);
      double s2 = fma(c[i + 2], e[i + 2], a2);
      double a3 = fma(c[i + 3], d[i + 3], s2);
      double s3 = fma(c[i + 3], e[i + 3], a3);
      a[i] = a0; b[i] = s0;
      a[i + 1] = a1; b[i + 1] = s1;
      a[i + 2] = a2; b[i + 2] = s2;
      a[i + 3] = a3; b[i + 3] = s3;
      s = s3;
    }
    for (; i < end; i++) {
      double ai = fma(c[i], d[i], s);
      s = fma(c[i], e[i], ai);
      a[i] = ai;
      b[i] = s;
    }
    sums[t] = s;
  }

  {
    double acc = b[0];
    for (int64_t t = 0; t < nblk; t++) {
      double st = sums[t];
      sums[t] = acc;
      acc += st;
    }
  }

  /* Phase 2: in-place offset fixup, vectorized (one offset per thread block) */
  #pragma omp parallel for num_threads(nt) schedule(static)
  for (int64_t t = 0; t < nblk; t++) {
    const int64_t s0 = 1 + t * S;
    const int64_t s1 = s0 + S < n ? s0 + S : n;
    const double off = sums[t];
    for (int64_t i = s0; i < s1; i++) {
      a[i] += off;
      b[i] += off;
    }
  }

  free(sums);
}
