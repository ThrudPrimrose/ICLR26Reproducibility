#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <stddef.h>

/* TSVC tsvc_2 s232:
 *   for j in 1..LEN_2D-1:
 *     for i in 1..j:
 *       aa[j,i] = aa[j,i-1]*aa[j,i-1] + bb[j,i]
 *
 * Each row j is an independent serial FMA chain (aa read is register-carried);
 * rows are partitioned across threads with exact workload balance
 * (row j costs j; prefix P(k)=k(k+1)/2), two rows interleaved per thread to
 * hide FMA latency, 16-byte non-temporal stores for the output.
 */

typedef __attribute__((vector_size(16))) double v2d;

static inline int64_t last_row_with_prefix_le(int64_t w) {
  /* largest k >= 0 with k*(k+1)/2 <= w */
  if (w <= 0) return 0;
  int64_t k = (int64_t)(sqrt(2.0 * (double)w + 0.25) - 0.5);
  while (k * (k + 1) / 2 > w) --k;
  while ((k + 1) * (k + 2) / 2 <= w) ++k;
  return k;
}

static inline void nt8(double *p, double v) {
  __asm__ volatile("movntsd %1, %0" : "=m"(*(double*)p) : "xmdf"(v));
}
static inline void nt16(double *p, double lo, double hi) {
  v2d v = { lo, hi };
  __asm__ volatile("movntdq %0, (%1)" : : "x"(v), "r"(p));
}

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  if (LEN_2D <= 1) return;
  const int64_t nrows = LEN_2D - 1;
  const int64_t total = nrows * (nrows + 1) / 2;

  int nt = omp_get_max_threads();
  if (nt < 1) nt = 1;
  if ((int64_t)nt > nrows) nt = (int)nrows;

  if ((int64_t)nt <= 1 || total < 100000) {
    for (int64_t j = 1; j < LEN_2D; ++j) {
      double *a = aa + j * LEN_2D;
      const double *b = bb + j * LEN_2D;
      double x = a[0];
      for (int64_t i = 1; i <= j; ++i) {
        x = x * x + b[i];
        a[i] = x;
      }
    }
    return;
  }

  #pragma omp parallel num_threads(nt)
  {
    const int64_t T = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t wlo = total * tid / T;
    const int64_t whi = total * (tid + 1) / T;
    /* unique owner: row j belongs to the thread with wlo <= P(j) < whi;
       the final row (P(j) = total) is kept by the last thread */
    const int64_t j0 = last_row_with_prefix_le(wlo - 1) + 1;
    const int64_t j1 = (tid == T - 1) ? nrows : last_row_with_prefix_le(whi - 1);

    /* interleave consecutive row pairs to hide FMA latency */
    int64_t jA = j0, jB = j0 + 1;
    for (; jB <= j1; jA += 2, jB += 2) {
      double *aA = aa + jA * LEN_2D;
      const double *bA = bb + jA * LEN_2D;
      double *aB = aa + jB * LEN_2D;
      const double *bB = bb + jB * LEN_2D;
      const int64_t nA = jA, nB = jB;
      double xA = aA[0], xB = aB[0];
      if (((uintptr_t)aA & 15) == 0 && ((uintptr_t)aB & 15) == 0) {
        /* 16B-aligned rows: 8B first element, then 16B NT stores of pairs */
        xA = xA * xA + bA[1]; nt8(aA + 1, xA);
        xB = xB * xB + bB[1]; nt8(aB + 1, xB);
        for (int64_t k = 2; k + 1 <= nB; k += 2) {
          if (k + 1 <= nA) {
            double t = xA * xA + bA[k];
            xA = t * t + bA[k + 1];
            nt16(aA + k, t, xA);
          }
          double t = xB * xB + bB[k];
          xB = t * t + bB[k + 1];
          nt16(aB + k, t, xB);
        }
        if ((nA & 1) == 0) { xA = xA * xA + bA[nA]; nt8(aA + nA, xA); }
        if ((nB & 1) == 0) { xB = xB * xB + bB[nB]; nt8(aB + nB, xB); }
      } else {
        /* unaligned rows: scalar 8B NT stores */
        int64_t i = 1;
        for (; i <= nA; ++i) {
          xA = xA * xA + bA[i]; nt8(aA + i, xA);
          xB = xB * xB + bB[i]; nt8(aB + i, xB);
        }
        for (; i <= nB; ++i) { xB = xB * xB + bB[i]; nt8(aB + i, xB); }
      }
    }
    if (jA <= j1) {
      /* final unpaired row */
      double *aA = aa + jA * LEN_2D;
      const double *bA = bb + jA * LEN_2D;
      double xA = aA[0];
      if (((uintptr_t)aA & 15) == 0) {
        xA = xA * xA + bA[1]; nt8(aA + 1, xA);
        int64_t k = 2;
        for (; k + 1 <= jA; k += 2) {
          double t = xA * xA + bA[k];
          xA = t * t + bA[k + 1];
          nt16(aA + k, t, xA);
        }
        if (k <= jA) { xA = xA * xA + bA[k]; nt8(aA + k, xA); }
      } else {
        for (int64_t i = 1; i <= jA; ++i) {
          xA = xA * xA + bA[i]; nt8(aA + i, xA);
        }
      }
    }
    __asm__ volatile("sfence" ::: "memory");
  }
}
