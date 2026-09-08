#include <stdint.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
static double hsum512(__m512d v) {
  __m256d lo = _mm512_castpd512_pd256(v);
  __m256d hi = _mm512_extractf64x4_pd(v, 1);
  __m256d s = _mm256_add_pd(lo, hi);
  __m128d a = _mm256_castpd256_pd128(s);
  __m128d b = _mm256_extractf128_pd(s, 1);
  __m128d t = _mm_add_pd(a, b);
  t = _mm_add_sd(t, _mm_unpackhi_pd(t, t));
  return _mm_cvtsd_f64(t);
}
static double blocks512(const double *restrict a, int64_t b0, int64_t b1) {
  __m512d av = _mm512_setzero_pd();
  const double *p = a + 8 * b0;
  const double *end = a + 8 * b1;
  while (p < end) {
    __m512d v = _mm512_loadu_pd(p);
    p += 8;
    av = _mm512_mask_add_pd(av, 170, av, v); /* lanes 1,3,5,7 = odd positions */
  }
  return hsum512(av);
}
#endif

#ifdef __AVX2__
#ifndef __AVX512F__
/* two 128-bit loads per 4 elements; lane 0 of each holds an odd element */
static double blocks256(const double *restrict a, int64_t b0, int64_t b1) {
  __m128d s0 = _mm_setzero_pd();
  __m128d s1 = _mm_setzero_pd();
  const double *p = a + 4 * b0;
  const double *end = a + 4 * b1;
  while (p < end) {
    s0 = _mm_add_pd(s0, _mm_loadu_pd(p + 1));
    s1 = _mm_add_pd(s1, _mm_loadu_pd(p + 3));
    p += 4;
  }
  double r0 = _mm_cvtsd_f64(s0); /* lane 0 = odd elements only */
  double r1 = _mm_cvtsd_f64(s1);
  return r0 + r1;
}
#endif
#endif

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  double acc = 0.0;
#ifdef __AVX512F__
  {
    const int64_t nblocks = LEN_1D / 8;
    if (LEN_1D >= 524288) {
      #pragma omp parallel reduction(+:acc)
      {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t per = nblocks / nt;
        int64_t rem = nblocks % nt;
        int64_t b0 = (int64_t)tid * per + (tid < rem ? tid : rem);
        int64_t b1 = b0 + per + (tid < rem ? 1 : 0);
        acc += blocks512(a, b0, b1);
      }
    } else {
      acc += blocks512(a, 0, nblocks);
    }
    for (int64_t i = nblocks * 8 + 1; i < LEN_1D; i += 2) acc += a[i];
  }
#elif defined(__AVX2__)
  {
    const int64_t nb4 = LEN_1D / 4;
    #pragma omp parallel reduction(+:acc)
    {
      int nt = omp_get_num_threads();
      int tid = omp_get_thread_num();
      int64_t per = nb4 / nt;
      int64_t rem = nb4 % nt;
      int64_t b0 = (int64_t)tid * per + (tid < rem ? tid : rem);
      int64_t b1 = b0 + per + (tid < rem ? 1 : 0);
      acc += blocks256(a, b0, b1);
    }
    for (int64_t i = nb4 * 4 + 1; i < LEN_1D; i += 2) acc += a[i];
  }
#else
  #pragma omp parallel for reduction(+:acc) schedule(static)
  for (int64_t i = 1; i < LEN_1D; i += 2) acc += a[i];
#endif
  out[0] = acc;
}
