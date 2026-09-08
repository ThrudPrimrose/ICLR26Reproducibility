/* TSVC s3111: sum of positive elements of a.
 * Hand-tuned AVX-512 inner loop (8 loads / 4 accumulators, masked select,
 * no branches) with chunked OpenMP parallel reduction. */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#if defined(__AVX512F__)

__attribute__((target("avx512f")))
static double sum_pos_avx512(const double *restrict p, int64_t m) {
  __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd(),
          a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
  const __m512d zero = _mm512_setzero_pd();
  int64_t i = 0;
  for (; i + 64 <= m; i += 64) {
    __m512d v0 = _mm512_loadu_pd(p + i);
    __m512d v1 = _mm512_loadu_pd(p + i + 8);
    __m512d v2 = _mm512_loadu_pd(p + i + 16);
    __m512d v3 = _mm512_loadu_pd(p + i + 24);
    __m512d v4 = _mm512_loadu_pd(p + i + 32);
    __m512d v5 = _mm512_loadu_pd(p + i + 40);
    __m512d v6 = _mm512_loadu_pd(p + i + 48);
    __m512d v7 = _mm512_loadu_pd(p + i + 56);
    a0 = _mm512_add_pd(a0, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v0, zero, _CMP_GT_OQ), v0));
    a1 = _mm512_add_pd(a1, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v1, zero, _CMP_GT_OQ), v1));
    a2 = _mm512_add_pd(a2, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v2, zero, _CMP_GT_OQ), v2));
    a3 = _mm512_add_pd(a3, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v3, zero, _CMP_GT_OQ), v3));
    a0 = _mm512_add_pd(a0, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v4, zero, _CMP_GT_OQ), v4));
    a1 = _mm512_add_pd(a1, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v5, zero, _CMP_GT_OQ), v5));
    a2 = _mm512_add_pd(a2, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v6, zero, _CMP_GT_OQ), v6));
    a3 = _mm512_add_pd(a3, _mm512_maskz_mov_pd(_mm512_cmp_pd_mask(v7, zero, _CMP_GT_OQ), v7));
  }
  __m512d acc = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
  double s = _mm512_reduce_add_pd(acc);
  for (; i < m; ++i)
    s += (p[i] > 0.0) * p[i];
  return s;
}

static double sum_pos(const double *p, int64_t m) { return sum_pos_avx512(p, m); }

#else

static double sum_pos(const double *p, int64_t m) {
  double s = 0.0;
  for (int64_t i = 0; i < m; ++i)
    s += (p[i] > 0.0) * p[i];
  return s;
}

#endif

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  double sum = 0.0;
  #pragma omp parallel reduction(+:sum)
  {
    const int tid = omp_get_thread_num();
    const int nt  = omp_get_num_threads();
    const int64_t chunk = (n + nt - 1) / nt;
    int64_t lo = (int64_t)tid * chunk;
    int64_t hi = lo + chunk;
    if (hi > n) hi = n;
    sum += sum_pos(a + lo, hi - lo);
  }
  b[0] = sum;
}
