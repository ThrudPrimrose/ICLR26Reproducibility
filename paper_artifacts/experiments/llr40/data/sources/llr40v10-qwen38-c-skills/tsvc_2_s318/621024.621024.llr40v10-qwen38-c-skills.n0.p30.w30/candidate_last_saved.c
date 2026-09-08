/* TSVC s318 v2: unroll-4 MLP on the AVX-512 inc==1 path. */
#include <math.h>
#include <stdint.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

typedef struct { double v; int64_t idx; } Mx;

static inline Mx mx_pick(Mx a, Mx b) {
  if (b.v > a.v) return b;
  if (b.v == a.v && b.idx < a.idx) return b;
  return a;
}

#if HAVE_AVX512
/* A has some lane > *bv (mask already checked, possibly stale after a prior
 * record in this unrolled group): take the vector max, keep first occurrence. */
static inline void rec(__m512d A, int64_t base, double *bv, int64_t *bi, __m512d *bvv) {
  double tmp[8];
  _mm512_storeu_pd(tmp, A);
  double v = tmp[0];
  int l = 0;
  for (int j = 1; j < 8; ++j)
    if (tmp[j] > v) { v = tmp[j]; l = j; }
  if (v > *bv) { *bv = v; *bi = base + l; *bvv = _mm512_set1_pd(v); }
}
#endif

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
  if (LEN_1D <= 0) { result[0] = 0.0; return; }

  const int nt = 2 * omp_get_max_threads();
  Mx part[nt > 256 ? 256 : nt > 0 ? nt : 1];
  for (int t = 0; t < nt; ++t) part[t] = (Mx) { -INFINITY, INT64_MAX };

  const int64_t span = (LEN_1D + nt - 1) / nt;

  #pragma omp parallel num_threads(nt)
  {
    const int64_t t = omp_get_thread_num();
    int64_t lo = t * span;
    int64_t hi = lo + span;
    if (hi > LEN_1D) hi = LEN_1D;

    double bv = -INFINITY;
    int64_t bi = INT64_MAX;

#if HAVE_AVX512
    if (inc == 1) {
      const double *p = a + lo;
      const int64_t n = hi - lo;
      int64_t k = 0;
      __m512d bvv = _mm512_set1_pd(bv);
      for (; k + 32 <= n; k += 32) {
        __m512d A0 = _mm512_loadu_pd(p + k);
        __m512d A1 = _mm512_loadu_pd(p + k + 8);
        __m512d A2 = _mm512_loadu_pd(p + k + 16);
        __m512d A3 = _mm512_loadu_pd(p + k + 24);
        A0 = _mm512_abs_pd(A0);
        A1 = _mm512_abs_pd(A1);
        A2 = _mm512_abs_pd(A2);
        A3 = _mm512_abs_pd(A3);
        if (_mm512_cmp_pd_mask(A0, bvv, _CMP_GT_OQ)) rec(A0, lo + k, &bv, &bi, &bvv);
        if (_mm512_cmp_pd_mask(A1, bvv, _CMP_GT_OQ)) rec(A1, lo + k + 8, &bv, &bi, &bvv);
        if (_mm512_cmp_pd_mask(A2, bvv, _CMP_GT_OQ)) rec(A2, lo + k + 16, &bv, &bi, &bvv);
        if (_mm512_cmp_pd_mask(A3, bvv, _CMP_GT_OQ)) rec(A3, lo + k + 24, &bv, &bi, &bvv);
      }
      for (; k + 8 <= n; k += 8) {
        __m512d A = _mm512_loadu_pd(p + k);
        A = _mm512_abs_pd(A);
        if (_mm512_cmp_pd_mask(A, bvv, _CMP_GT_OQ)) rec(A, lo + k, &bv, &bi, &bvv);
      }
      for (; k < n; ++k) {
        double v = fabs(p[k]);
        if (v > bv) { bv = v; bi = lo + k; }
      }
    } else
#endif
    {
      int64_t kk = lo * inc;
      for (int64_t i = lo; i < hi; ++i, kk += inc) {
        double v = fabs(a[kk]);
        if (v > bv) { bv = v; bi = i; }
      }
    }
    part[t] = (Mx) { bv, bi };
  }

  Mx g = mx_pick((Mx) { fabs(a[0]), 0 }, part[0]);
  for (int t = 1; t < nt; ++t) g = mx_pick(g, part[t]);
  result[0] = g.v + (double) g.idx;
}
