/* Optimized tsvc_2 s3110: row-major max + first-occurrence index of a 2D array.
 * Output: bb[0] = maxv + (double)xindex + (double)yindex  (xindex/yindex of the
 * FIRST occurrence of the global max, row-major).
 *
 * Strategy:
 *  - OpenMP: each thread scans a contiguous row-major chunk, reducing to
 *    (block-max, first-position-in-block).
 *  - Combine: max by value, ties by smallest position (strictly associative).
 *  - Vector scan with AVX-512: 16 doubles/iter, running max broadcast compare,
 *    first-strict-improvement lane gives the position (rare branch).
 *  - NaN semantics match the reference (strict > : NaN never updates max;
 *    if aa[0] is NaN the reference keeps NaN at (0,0) forever).
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
static int diag_done = 0;

typedef struct { double v; int64_t p; int have; } Cand;

static Cand scan_block(const double *restrict a, int64_t n, int64_t pos0) {
  double m = -INFINITY;          /* running max; -inf == "no value seen yet" */
  int64_t p = pos0;              /* first position attaining m within [pos0,pos0+n) */
  int have = 0;
  int64_t i = 0;
  const int64_t n16 = n & ~(int64_t)15;
  for (; i < n16; i += 16) {
    const double *ap = a + i;
    __m512d bm = _mm512_set1_pd(m);
    __m512d x1 = _mm512_loadu_pd(ap);
    __m512d x2 = _mm512_loadu_pd(ap + 8);
    __mmask8 g1 = _mm512_cmp_pd_mask(x1, bm, _CMP_GT_OQ);
    __mmask8 g2 = _mm512_cmp_pd_mask(x2, bm, _CMP_GT_OQ);
    unsigned g = (unsigned)g1 | ((unsigned)g2 << 8);
    if (g) {
      /* block max beats running max: m becomes the block max; its first
         occurrence in this block is the first lane equal to it */
      __m512d w = _mm512_max_pd(_mm512_max_pd(bm, x1), x2);
      double tmp[8];
      _mm512_storeu_pd(tmp, w);
      double u0 = tmp[0], u1 = tmp[1], u2 = tmp[2], u3 = tmp[3];
      double u4 = tmp[4], u5 = tmp[5], u6 = tmp[6], u7 = tmp[7];
      if (u1 > u0) u0 = u1;
      if (u2 > u0) u0 = u2;
      if (u3 > u0) u0 = u3;
      if (u5 > u4) u4 = u5;
      if (u6 > u4) u4 = u6;
      if (u7 > u4) u4 = u7;
      if (u4 > u0) u0 = u4;
      m = u0;
      __mmask8 e1 = _mm512_cmp_pd_mask(x1, _mm512_set1_pd(m), _CMP_EQ_OQ);
      __mmask8 e2 = _mm512_cmp_pd_mask(x2, _mm512_set1_pd(m), _CMP_EQ_OQ);
      unsigned e = (unsigned)e1 | ((unsigned)e2 << 8);
      p = pos0 + i + __builtin_ctz(e);
      have = 1;
    }
  }
  for (; i < n; ++i) {
    double v = a[i];
    if (v > m) { m = v; p = pos0 + i; have = 1; }
  }
  return (Cand){m, p, have};
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb,
                       const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 0) { bb[0] = 0.0; return; }
  const int64_t total = N * N;

  if (!diag_done) {
    diag_done = 1;
    struct timespec a, b;
    tsvc_2_s3110_fp64(aa, bb, LEN_2D); /* warm up first (recursion-safe: guard set) */
    clock_gettime(CLOCK_MONOTONIC, &a);
    tsvc_2_s3110_fp64(aa, bb, LEN_2D);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ms1 = (b.tv_sec - a.tv_sec)*1000.0 + (b.tv_nsec - a.tv_nsec)/1e6;
    tsvc_2_s3110_fp64(aa, bb, LEN_2D);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ms2 = (b.tv_sec - a.tv_sec)*1000.0 + (b.tv_nsec - a.tv_nsec)/1e6;
    printf("DIAG N=%ld total_bytes=%ld max_threads=%d num_procs=%d run1_ms=%.3f run2_ms=%.3f gbps1=%.2f gbps2=%.2f\n",
           (long)LEN_2D, (long)(total*8), omp_get_max_threads(), omp_get_num_procs(), ms1, ms2,
           total*8/ms1/1e6, total*8/ms2/1e6);
    fflush(stdout);
  }

  /* Reference: maxv starts as aa[0]; if it is NaN, nothing ever compares
     greater, so the answer is NaN at (0,0). */
  if (isnan(aa[0])) { bb[0] = aa[0] + 0.0; return; }

  const int64_t PAR_MIN = 1 << 18; /* 256K doubles (2 MB): use threads above this */
  Cand cands[512];
  int nt = 1;

  if (total >= PAR_MIN) {
    nt = omp_get_max_threads();
    if (nt > 512) nt = 512;
    if (nt < 1) nt = 1;
    const int64_t base = total / nt;
    const int64_t rem = total % nt;
    #pragma omp parallel num_threads(nt)
    {
      const int t = omp_get_thread_num();
      int64_t start = base * t + (t < rem ? t : rem);
      int64_t end = base * (t + 1) + (t + 1 < rem ? t + 1 : rem);
      cands[t] = scan_block(aa + start, end - start, start);
    }
  } else {
    cands[0] = scan_block(aa, total, 0);
  }

  double bv = 0.0;
  int64_t bp = 0;
  int have = 0;
  for (int t = 0; t < nt; ++t) {
    const Cand *c = &cands[t];
    if (!c->have) continue;
    if (!have) { have = 1; bv = c->v; bp = c->p; }
    else if (c->v > bv) { bv = c->v; bp = c->p; }
    else if (c->v == bv && c->p < bp) { bp = c->p; }
  }

  if (!have) { bb[0] = -INFINITY + 0.0; return; } /* all elements -inf */
  int64_t xindex = bp / N;
  int64_t yindex = bp - xindex * N;
  /* two separate roundings, exactly as the numpy oracle ((maxv + x) + y) */
  double chksum = bv + (double)xindex;
  chksum = chksum + (double)yindex;
  bb[0] = chksum;
}
