/* Sum of positive elements: b[0] = sum(a[i] for a[i] > 0).
 *
 * OpenMP-parallel: each thread owns a contiguous slice and sums it in order,
 * so the total differs from the sequential reference only by reordering at
 * slice boundaries (far below rtol). AVX-512 masked adds with four in-order
 * accumulators; SSE2/scalar fallbacks keep the same structure.
 */
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
static inline double slice_avx512(const double *p, int64_t len) {
  __m512d v0 = _mm512_setzero_pd(), v1 = _mm512_setzero_pd(),
          v2 = _mm512_setzero_pd(), v3 = _mm512_setzero_pd();
  __m512d vz = _mm512_setzero_pd();
  int64_t i = 0;
  for (; i + 32 <= len; i += 32) {
    __m512d x0 = _mm512_loadu_pd(p + i);
    __m512d x1 = _mm512_loadu_pd(p + i + 8);
    __m512d x2 = _mm512_loadu_pd(p + i + 16);
    __m512d x3 = _mm512_loadu_pd(p + i + 24);
    v0 = _mm512_mask_add_pd(v0, _mm512_cmp_pd_mask(x0, vz, _CMP_GT_OQ), v0, x0);
    v1 = _mm512_mask_add_pd(v1, _mm512_cmp_pd_mask(x1, vz, _CMP_GT_OQ), v1, x1);
    v2 = _mm512_mask_add_pd(v2, _mm512_cmp_pd_mask(x2, vz, _CMP_GT_OQ), v2, x2);
    v3 = _mm512_mask_add_pd(v3, _mm512_cmp_pd_mask(x3, vz, _CMP_GT_OQ), v3, x3);
  }
  double s = _mm512_reduce_add_pd(v0) + _mm512_reduce_add_pd(v1) +
             _mm512_reduce_add_pd(v2) + _mm512_reduce_add_pd(v3);
  for (; i < len; ++i)
    if (p[i] > 0.0) s += p[i];
  return s;
}
#endif

#if !defined(__AVX512F__) && defined(__SSE2__)
static inline double slice_sse(const double *p, int64_t len) {
  __m128d v0 = _mm_setzero_pd(), v1 = _mm_setzero_pd();
  __m128d vz = _mm_setzero_pd();
  int64_t i = 0;
  for (; i + 8 <= len; i += 8) {
    __m128d x0 = _mm_loadu_pd(p + i);
    __m128d x1 = _mm_loadu_pd(p + i + 4);
    v0 = _mm_add_pd(v0, _mm_and_pd(_mm_cmpgt_pd(x0, vz), x0));
    v1 = _mm_add_pd(v1, _mm_and_pd(_mm_cmpgt_pd(x1, vz), x1));
  }
  double s = _mm_cvtsd_f64(v0) + _mm_cvtsd_f64(_mm_unpackhi_pd(v0, v0)) +
             _mm_cvtsd_f64(v1) + _mm_cvtsd_f64(_mm_unpackhi_pd(v1, v1));
  for (; i < len; ++i)
    if (p[i] > 0.0) s += p[i];
  return s;
}
#endif

static inline double slice_scalar(const double *p, int64_t len) {
  double s = 0.0;
  for (int64_t i = 0; i < len; ++i)
    if (p[i] > 0.0) s += p[i];
  return s;
}

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  double sum = 0.0;
#ifdef _OPENMP
  #pragma omp parallel reduction(+:sum)
#endif
  {
    int tid, t;
#ifdef _OPENMP
    tid = omp_get_thread_num();
    t = omp_get_num_threads();
#else
    tid = 0;
    t = 1;
#endif
    int64_t lo = (LEN_1D / t) * tid;
    int64_t hi = tid == t - 1 ? LEN_1D : (LEN_1D / t) * (tid + 1);
    double s;
#if defined(__AVX512F__)
    s = slice_avx512(a + lo, hi - lo);
#elif defined(__SSE2__)
    s = slice_sse(a + lo, hi - lo);
#else
    s = slice_scalar(a + lo, hi - lo);
#endif
    sum += s;
  }
  b[0] = sum;
}
