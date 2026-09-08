#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/*
 * tsvc_2 s323:
 *   a[i] = b[i-1] + c[i]*d[i]
 *   b[i] = a[i]   + c[i]*e[i]
 * The loop-carried dependence is a prefix sum:
 *   u[i] = c[i]*(d[i]+e[i])
 *   b[i] = b[0] + sum_{j=1..i} u[j]
 *   a[i] = b[i] - c[i]*e[i]
 *
 * Two-level parallel scan on the target:
 *  Phase 1: per-block serial prefix sums -> total1[k]     (parallel over blocks)
 *  Phase 2: per-superblock sums of total1 -> total2[s]    (parallel over superblocks)
 *  Phase 3: exclusive prefix of total2 -> pref2[s]        (serial, <= nblk/1024 steps)
 *  Phase 4: re-scan each block to finalize b[i], a[i]     (parallel over blocks)
 * All arithmetic inside a block is in the exact serial order; only the
 * cross-block reassociation differs from the reference (scan band).
 */
#define S323_T1 1024
#define S323_T2 1024

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  const int64_t m = LEN_1D - 1;
  if (m <= 0) return;
  const int64_t nblk = (m + S323_T1 - 1) / S323_T1;
  const int64_t nsup = (nblk + S323_T2 - 1) / S323_T2;
  const double b0 = b[0];
  double *const total1 = malloc((size_t)nblk * sizeof(double));
  double *const total2 = malloc((size_t)nsup * sizeof(double));
  double *const pref2 = malloc((size_t)nsup * sizeof(double));

  #pragma omp target data map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
                          map(from: a[1:LEN_1D], b[1:LEN_1D]) \
                          map(to: total1[0:nblk], total2[0:nsup], pref2[0:nsup])
  {
    #pragma omp target teams distribute parallel for
    for (int64_t k = 0; k < nblk; k++) {
      const int64_t beg = k * S323_T1 + 1;
      const int64_t end = beg + S323_T1 < LEN_1D ? beg + S323_T1 : LEN_1D;
      double s = 0.0;
      for (int64_t i = beg; i < end; i++) s += c[i] * (d[i] + e[i]);
      total1[k] = s;
    }

    #pragma omp target teams distribute parallel for
    for (int64_t s2 = 0; s2 < nsup; s2++) {
      const int64_t beg = s2 * S323_T2;
      const int64_t end = beg + S323_T2 < nblk ? beg + S323_T2 : nblk;
      double s = 0.0;
      for (int64_t k = beg; k < end; k++) s += total1[k];
      total2[s2] = s;
    }

    #pragma omp target
    {
      double acc = 0.0;
      for (int64_t s2 = 0; s2 < nsup; s2++) {
        pref2[s2] = acc;
        acc += total2[s2];
      }
    }

    #pragma omp target teams distribute parallel for
    for (int64_t k = 0; k < nblk; k++) {
      const int64_t beg = k * S323_T1 + 1;
      const int64_t end = beg + S323_T1 < LEN_1D ? beg + S323_T1 : LEN_1D;
      double bpref = pref2[k / S323_T2];
      int64_t t = (k / S323_T2) * S323_T2;
      while (t < k) {
        bpref += total1[t];
        t++;
      }
      double s = 0.0;
      for (int64_t i = beg; i < end; i++) {
        s += c[i] * (d[i] + e[i]);
        b[i] = b0 + bpref + s;
        a[i] = b[i] - c[i] * e[i];
      }
    }
  }
  free(total1);
  free(total2);
  free(pref2);
}
