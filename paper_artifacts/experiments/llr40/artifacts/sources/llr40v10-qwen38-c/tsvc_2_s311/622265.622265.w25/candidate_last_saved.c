/* Optimized TSVC s311: parallel multi-accumulator SIMD sum (fp64). */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static inline double tsvc_2_s311_simd_sum(const double *restrict a, int64_t n) {
#if defined(__AVX512F__)
  __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd(),
          s2 = _mm512_setzero_pd(), s3 = _mm512_setzero_pd(),
          s4 = _mm512_setzero_pd(), s5 = _mm512_setzero_pd(),
          s6 = _mm512_setzero_pd(), s7 = _mm512_setzero_pd();
  int64_t i = 0;
  const int64_t stride = 64;
  int64_t n8 = n & ~(stride - 1);
  for (; i < n8; i += stride) {
    __m512d l0 = _mm512_loadu_pd(a + i);
    __m512d l1 = _mm512_loadu_pd(a + i + 8);
    __m512d l2 = _mm512_loadu_pd(a + i + 16);
    __m512d l3 = _mm512_loadu_pd(a + i + 24);
    __m512d l4 = _mm512_loadu_pd(a + i + 32);
    __m512d l5 = _mm512_loadu_pd(a + i + 40);
    __m512d l6 = _mm512_loadu_pd(a + i + 48);
    __m512d l7 = _mm512_loadu_pd(a + i + 56);
    s0 = _mm512_add_pd(s0, l0);
    s1 = _mm512_add_pd(s1, l1);
    s2 = _mm512_add_pd(s2, l2);
    s3 = _mm512_add_pd(s3, l3);
    s4 = _mm512_add_pd(s4, l4);
    s5 = _mm512_add_pd(s5, l5);
    s6 = _mm512_add_pd(s6, l6);
    s7 = _mm512_add_pd(s7, l7);
  }
  double t = 0.0;
  for (; i < n; i++) t += a[i];
  {
    __m512d s = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(s0, s1),
                                            _mm512_add_pd(s2, s3)),
                              _mm512_add_pd(_mm512_add_pd(s4, s5),
                                           _mm512_add_pd(s6, s7)));
    return _mm512_reduce_add_pd(s) + t;
  }
#else
  double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0;
  int64_t i = 0, n4 = n & ~3;
  for (; i < n4; i += 4) {
    p0 += a[i]; p1 += a[i + 1]; p2 += a[i + 2]; p3 += a[i + 3];
  }
  double t = 0.0;
  for (; i < n; i++) t += a[i];
  return (p0 + p1) + (p2 + p3) + t;
#endif
}

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out,
                      const int64_t LEN_1D) {
  if (LEN_1D <= 0) { sum_out[0] = 0.0; return; }
  if (LEN_1D < (1 << 18)) {
    sum_out[0] = tsvc_2_s311_simd_sum(a, LEN_1D);
    return;
  }
  double total = 0.0;
#pragma omp parallel reduction(+:total)
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t base = (LEN_1D / nt) * tid;
    int64_t end = (tid == nt - 1) ? LEN_1D : (LEN_1D / nt) * (tid + 1);
    total += tsvc_2_s311_simd_sum(a + base, end - base);
  }
  sum_out[0] = total;
}
