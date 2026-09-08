#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

static inline void kahan512(__m512d *s, __m512d *c, __m512d x) {
  __m512d y = _mm512_sub_pd(x, *c);
  __m512d t = _mm512_add_pd(*s, y);
  *c = _mm512_sub_pd(_mm512_sub_pd(t, *s), y);
  *s = t;
}
#define KADD(S, C, x)                 \
  do {                                \
    double y__ = (x) - (C);           \
    double t__ = (S) + y__;           \
    (C) = (t__ - (S)) - y__;          \
    (S) = t__;                        \
  } while (0)

#define VECBODY(STORE_A, STORE_B, PREF)                                    \
  {                                                                         \
    int64_t al = (lo + 7) & ~(int64_t)7;                                    \
    int64_t lim = hi & ~(int64_t)7;                                         \
    int64_t i = lo;                                                         \
    for (; i < al && i < hi; ++i) {                                         \
      double va = c[i] + d[i];                                              \
      double vb = c[i] + e[i];                                              \
      a[i] = va;                                                            \
      b[i] = vb;                                                            \
      KADD(s, cc, va);                                                      \
      KADD(s, cc, vb);                                                      \
    }                                                                       \
    for (; i < lim; i += 8) {                                               \
      PREF;                                                                  \
      __m512d vc = _mm512_loadu_pd(c + i);                                  \
      __m512d vd = _mm512_loadu_pd(d + i);                                  \
      __m512d ve = _mm512_loadu_pd(e + i);                                  \
      __m512d va = _mm512_add_pd(vc, vd);                                   \
      __m512d vb = _mm512_add_pd(vc, ve);                                   \
      STORE_A;                                                               \
      STORE_B;                                                               \
      kahan512(&sa, &ca, va);                                                \
      kahan512(&sb, &cb, vb);                                                \
    }                                                                       \
    for (; i < hi; ++i) {                                                   \
      double va = c[i] + d[i];                                              \
      double vb = c[i] + e[i];                                              \
      a[i] = va;                                                            \
      b[i] = vb;                                                            \
      KADD(s, cc, va);                                                      \
      KADD(s, cc, vb);                                                      \
    }                                                                       \
  }

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  struct timespec pt0, pt1;
  clock_gettime(CLOCK_MONOTONIC, &pt0);
  int nt = omp_get_max_threads();
  int64_t ali = ((((uintptr_t)a) | ((uintptr_t)b)) & 63) == 0;
  if (nt > 1 && LEN_1D >= 16384) {
    double *parts = malloc((size_t)nt * 2 * sizeof(double));
    if (parts) {
      double *pcs = parts + nt;
#pragma omp parallel num_threads(nt)
      {
        int tid = omp_get_thread_num();
        int nt2 = omp_get_num_threads();
        int64_t base = LEN_1D / nt2, rem = LEN_1D % nt2;
        int64_t lo = tid * base + (tid < rem ? tid : rem);
        int64_t hi = lo + base + (tid < rem ? 1 : 0);
        __m512d sa = _mm512_setzero_pd(), ca = _mm512_setzero_pd();
        __m512d sb = _mm512_setzero_pd(), cb = _mm512_setzero_pd();
        double s = 0.0, cc = 0.0;
        if (ali)
          VECBODY(_mm512_stream_pd(a + i, va), _mm512_stream_pd(b + i, vb),
                  (void)0)
        else
          VECBODY(_mm512_storeu_pd(a + i, va), _mm512_storeu_pd(b + i, vb),
                  (void)0)
        double S = 0.0, C = 0.0;
        double xs[8];
        _mm512_storeu_pd(xs, sa);
        for (int l = 0; l < 8; ++l) KADD(S, C, xs[l]);
        _mm512_storeu_pd(xs, ca);
        for (int l = 0; l < 8; ++l) KADD(S, C, xs[l]);
        _mm512_storeu_pd(xs, sb);
        for (int l = 0; l < 8; ++l) KADD(S, C, xs[l]);
        _mm512_storeu_pd(xs, cb);
        for (int l = 0; l < 8; ++l) KADD(S, C, xs[l]);
        KADD(S, C, s);
        parts[tid] = S;
        pcs[tid] = C;
      }
      double S = 0.0, C = 0.0;
      for (int t = 0; t < nt; ++t) {
        KADD(S, C, parts[t]);
        KADD(S, C, pcs[t]);
      }
      free(parts);
      b[0] = S;
      clock_gettime(CLOCK_MONOTONIC, &pt1);
      fprintf(stdout, "PROBE-B2 NT-ONLY LEN=%lld NT=%d time=%.3fms rate=%.1fGB/s\n", (long long)LEN_1D, nt,
              ((pt1.tv_sec - pt0.tv_sec) * 1e3) + ((pt1.tv_nsec - pt0.tv_nsec) * 1e-6),
              5.0 * LEN_1D * 8 / ((pt1.tv_sec - pt0.tv_sec) + 1e-9 * (pt1.tv_nsec - pt0.tv_nsec)) / 1e9);
      fflush(stdout);
      return;
    }
    free(parts);
  }
  double sum = 0.0;
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] = c[i] + d[i];
    sum += a[i];
    b[i] = c[i] + e[i];
    sum += b[i];
  }
  b[0] = sum;
  clock_gettime(CLOCK_MONOTONIC, &pt1);
  fprintf(stdout, "PROBE-B-serial LEN=%lld time=%.3fms\n", (long long)LEN_1D,
          ((pt1.tv_sec - pt0.tv_sec) * 1e3) + ((pt1.tv_nsec - pt0.tv_nsec) * 1e-6));
  fflush(stdout);
}
