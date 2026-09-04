#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* Prefix sum (scan): b[i] = a[0] + a[1] + ... + a[i].
 *
 * Single-pass SIMD scan over 8-double (512-bit) blocks.  Within a block an
 * inclusive scan is built with a 1-2-4 tree (reassociated vs the serial
 * order); the carry between blocks keeps the exact serial structure
 * c_{m+1} = c_m + block_total_m.  Blocks are split across OpenMP threads.
 */

#if defined(__AVX512F__)

static const int64_t I1_[8] = {0, 0, 1, 2, 3, 4, 5, 6}; /* shift right 1 */
static const int64_t I2_[8] = {0, 1, 0, 1, 2, 3, 4, 5}; /* shift right 2 */
static const int64_t I4_[8] = {0, 1, 2, 3, 0, 1, 2, 3}; /* shift right 4 */
static const int64_t I7_[8] = {7, 7, 7, 7, 7, 7, 7, 7}; /* lane 7 */


#endif

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D)
{
  const int64_t N = LEN_1D;
  if (N <= 0)
    return;

  int nt = 8;
  int mt = omp_get_max_threads();
  if (mt < nt)
    nt = mt;
  while (nt > 1 && N / nt < 1024)
    nt--;

  const int64_t chunk = (N + nt - 1) / nt;

  #pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    int64_t lo = (int64_t)tid * chunk;
    int64_t hi = lo + chunk;
    if (hi > N)
      hi = N;
    const double *p = a + lo;
    double *q = b + lo;
    int64_t n = hi - lo;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512i I1 = _mm512_loadu_si512((const __m512i *)I1_);
    __m512i I2 = _mm512_loadu_si512((const __m512i *)I2_);
    __m512i I4 = _mm512_loadu_si512((const __m512i *)I4_);
    __m512i I7 = _mm512_loadu_si512((const __m512i *)I7_);
    #define SCAN8(V, R, C)                                          \
      {                                                             \
        __m512d t = (V);                                            \
        t = _mm512_mask_add_pd(t, 0xFE, t, _mm512_permutexvar_pd(I1, t)); \
        t = _mm512_mask_add_pd(t, 0xFC, t, _mm512_permutexvar_pd(I2, t)); \
        t = _mm512_mask_add_pd(t, 0xF0, t, _mm512_permutexvar_pd(I4, t)); \
        (R) = _mm512_add_pd(t, _mm512_set1_pd(C));                  \
        (C) += _mm512_cvtsd_f64(_mm512_permutexvar_pd(I7, t));      \
      }
    double c = 0.0;
    for (; i + 32 <= n; i += 32) {
      __m512d v0 = _mm512_loadu_pd(p + i);
      __m512d v1 = _mm512_loadu_pd(p + i + 8);
      __m512d v2 = _mm512_loadu_pd(p + i + 16);
      __m512d v3 = _mm512_loadu_pd(p + i + 24);
      __m512d r0, r1, r2, r3;
      SCAN8(v0, r0, c);
      _mm512_storeu_pd(q + i, r0);
      SCAN8(v1, r1, c);
      _mm512_storeu_pd(q + i + 8, r1);
      SCAN8(v2, r2, c);
      _mm512_storeu_pd(q + i + 16, r2);
      SCAN8(v3, r3, c);
      _mm512_storeu_pd(q + i + 24, r3);
    }
    for (; i + 8 <= n; i += 8) {
      __m512d v = _mm512_loadu_pd(p + i);
      __m512d r;
      SCAN8(v, r, c);
      _mm512_storeu_pd(q + i, r);
    }
#undef SCAN8
#endif
    for (; i < n; i++) {
      c += p[i];
      q[i] = c;
    }
  }
}
