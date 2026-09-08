#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#if defined(__AVX512F__)
#define HAS_V512 1
#else
#define HAS_V512 0
#endif

static inline double red512(__m512d v) {
  __m256d v0 = _mm512_extractf64x4_pd(v, 0);
  __m256d v1 = _mm512_extractf64x4_pd(v, 1);
  v0 = _mm256_add_pd(v0, v1);
  __m128d lo = _mm256_extractf128_pd(v0, 0);
  __m128d hi = _mm256_extractf128_pd(v0, 1);
  lo = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
}

/* Permute-index layout for _mm512_permutex2var_pd is probed at runtime:
   mode 0: lane l selects index word l   (documented behavior)
   mode 1: lane l selects index word 2*l (observed on this AMD CPU) */
static int g_perm2_mode = 2; /* 2 = uncalibrated */

static void calibrate_perm2(void) {
  if (g_perm2_mode != 2) return;
  const int idx[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  const double aa[8] = {1,2,3,4,5,6,7,8};
  const double bb[8] = {-1,-2,-3,-4,-5,-6,-7,-8};
  __m512d p = _mm512_permutex2var_pd(_mm512_loadu_pd(aa),
                                     _mm512_loadu_si512(idx),
                                     _mm512_loadu_pd(bb));
  double o[8];
  _mm512_storeu_pd(o, p);
  g_perm2_mode = (o[1] == 2.0) ? 0 : 1;
}

#if HAS_V512
void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  calibrate_perm2();

  int nt = (int)omp_get_max_threads();
  if (nt < 1) nt = 1;
  if (nt > 32) nt = 32;

  if (nt == 1 || LEN_1D < (1 << 20)) {
    double s = 0.0;
    for (int64_t i = 0; i < LEN_1D; ++i) { s += a[i]; b[i] = s; }
    return;
  }

  const int wide = (g_perm2_mode == 1);
  const __m512i sh1 = wide
    ? _mm512_setr_epi32(8,7,0,7,1,7,2,7,3,7,4,7,5,7,6,7)
    : _mm512_setr_epi32(8,0,1,2,3,4,5,6,7,7,7,7,7,7,7,7);
  const __m512i sh2 = wide
    ? _mm512_setr_epi32(8,7,9,7,0,7,1,7,2,7,3,7,4,7,5,7)
    : _mm512_setr_epi32(8,9,0,1,2,3,4,5,7,7,7,7,7,7,7,7);
  const __m512i sh4 = wide
    ? _mm512_setr_epi32(8,7,9,7,10,7,11,7,0,7,1,7,2,7,3,7)
    : _mm512_setr_epi32(8,9,10,11,0,1,2,3,7,7,7,7,7,7,7,7);
  const __m512i all7 = _mm512_setr_epi32(7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7);
  const __m512d vzero = _mm512_setzero_pd();

  double rsum[32], offs[33];
  const int64_t chunk = (LEN_1D + nt - 1) / nt;

#pragma omp parallel num_threads(nt)
  {
    const int t = omp_get_thread_num();
    const int64_t lo = (int64_t)t * chunk;
    int64_t hi = lo + chunk;
    if (hi > LEN_1D) hi = LEN_1D;
    int64_t i;

    /* pass 1: region sum, 4 Neumaier-compensated accumulators */
    {
      __m512d v0 = vzero, v1 = vzero, v2 = vzero, v3 = vzero;
      __m512d c0 = vzero, c1 = vzero, c2 = vzero, c3 = vzero;
      const int64_t vq = lo + (((hi - lo) >> 5) << 5);
      for (i = lo; i < vq; i += 32) {
        __m512d f, tt;
        __m512d va = _mm512_loadu_pd(a + i);
        f = _mm512_sub_pd(va, c0); tt = _mm512_add_pd(v0, f);
        c0 = _mm512_sub_pd(_mm512_sub_pd(tt, v0), f); v0 = tt;
        va = _mm512_loadu_pd(a + i + 8);
        f = _mm512_sub_pd(va, c1); tt = _mm512_add_pd(v1, f);
        c1 = _mm512_sub_pd(_mm512_sub_pd(tt, v1), f); v1 = tt;
        va = _mm512_loadu_pd(a + i + 16);
        f = _mm512_sub_pd(va, c2); tt = _mm512_add_pd(v2, f);
        c2 = _mm512_sub_pd(_mm512_sub_pd(tt, v2), f); v2 = tt;
        va = _mm512_loadu_pd(a + i + 24);
        f = _mm512_sub_pd(va, c3); tt = _mm512_add_pd(v3, f);
        c3 = _mm512_sub_pd(_mm512_sub_pd(tt, v3), f); v3 = tt;
      }
      v0 = _mm512_add_pd(v0, v1);
      v2 = _mm512_add_pd(v2, v3);
      v0 = _mm512_add_pd(v0, v2);
      c0 = _mm512_add_pd(c0, c1);
      c2 = _mm512_add_pd(c2, c3);
      c0 = _mm512_add_pd(c0, c2);
      double s = red512(v0) + red512(c0);
      for (; i < hi; ++i) s += a[i];
      rsum[t] = s;
    }
#pragma omp barrier
    if (t == 0) {
      offs[0] = 0.0;
      for (int k = 0; k < nt; ++k) offs[k + 1] = offs[k] + rsum[k];
    }
#pragma omp barrier

    /* pass 2: vector scan with Kahan-compensated running total */
    {
      __m512d Sh = _mm512_set1_pd(offs[t]);
      __m512d Cv = vzero;
      const int64_t vb = lo + (((hi - lo) >> 3) << 3);
      for (i = lo; i < vb; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d t1 = _mm512_add_pd(va, _mm512_permutex2var_pd(va, sh1, vzero));
        t1 = _mm512_add_pd(t1, _mm512_permutex2var_pd(t1, sh2, vzero));
        t1 = _mm512_add_pd(t1, _mm512_permutex2var_pd(t1, sh4, vzero));
        __m512d t7b = _mm512_permutex2var_pd(t1, all7, t1);
        _mm512_storeu_pd(b + i, _mm512_add_pd(_mm512_add_pd(t1, Sh), Cv));
        __m512d y = _mm512_sub_pd(t7b, Cv);
        __m512d tt = _mm512_add_pd(Sh, y);
        Cv = _mm512_sub_pd(_mm512_sub_pd(tt, Sh), y);
        Sh = tt;
      }
      double S = _mm_cvtsd_f64(_mm256_extractf128_pd(_mm512_extractf64x4_pd(Sh, 0), 0))
              + _mm_cvtsd_f64(_mm256_extractf128_pd(_mm512_extractf64x4_pd(Cv, 0), 0));
      for (; i < hi; ++i) { S += a[i]; b[i] = S; }
    }
  }
}
#else
void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  double s = 0.0;
  for (int64_t i = 0; i < LEN_1D; ++i) { s += a[i]; b[i] = s; }
}
#endif
