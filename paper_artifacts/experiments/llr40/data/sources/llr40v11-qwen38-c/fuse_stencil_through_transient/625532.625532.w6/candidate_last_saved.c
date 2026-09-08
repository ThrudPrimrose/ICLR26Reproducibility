#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* TSVC fuse_stencil_through_transient:
 *   out[i] = (a[i-1]+a[i]+a[i+1]) * (a[i]+a[i+1]+a[i+2])   for 1 <= i <= LEN_1D-3
 *
 * Vector block of 8 (512-bit) starting at i0:
 *   lf = a[i0-1..i0+6], v0 = a[i0..i0+7], v1 = a[i0+1..i0+8], v2 = a[i0+2..i0+9]
 *   L = lf + v0 + v1    (a[i-1]+a[i]+a[i+1])
 *   R = v0 + v1 + v2    (a[i]+a[i+1]+a[i+2])
 *   out = L * R
 */
void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  if (LEN_1D < 4) return;
  const int64_t N = LEN_1D - 3; /* max output index */
#pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t lo = 1 + (N) * tid / nt;
    const int64_t hi = 1 + (N) * (tid + 1) / nt; /* i in [lo, hi) */
#if defined(__AVX512F__)
    {
      const int64_t vstart = (lo + 7) & ~7LL;
      for (int64_t i = lo; i < vstart && i < hi; ++i)
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
      int64_t i0 = vstart;
      for (; i0 + 7 < hi; i0 += 8) {
        __m512d lf = _mm512_loadu_pd(a + i0 - 1);
        __m512d v0 = _mm512_loadu_pd(a + i0);
        __m512d v1 = _mm512_loadu_pd(a + i0 + 1);
        __m512d v2 = _mm512_loadu_pd(a + i0 + 2);
        __m512d L = _mm512_add_pd(_mm512_add_pd(lf, v0), v1);
        __m512d R = _mm512_add_pd(_mm512_add_pd(v0, v1), v2);
        _mm512_storeu_pd(out + i0, _mm512_mul_pd(L, R));
      }
      for (int64_t i = i0; i < hi; ++i)
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
#elif defined(__AVX2__)
    {
      const int64_t vstart = (lo + 3) & ~3LL;
      for (int64_t i = lo; i < vstart && i < hi; ++i)
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
      int64_t i0 = vstart;
      for (; i0 + 3 < hi; i0 += 4) {
        __m256d lf = _mm256_loadu_pd(a + i0 - 1);
        __m256d v0 = _mm256_loadu_pd(a + i0);
        __m256d v1 = _mm256_loadu_pd(a + i0 + 1);
        __m256d v2 = _mm256_loadu_pd(a + i0 + 2);
        __m256d L = _mm256_add_pd(_mm256_add_pd(lf, v0), v1);
        __m256d R = _mm256_add_pd(_mm256_add_pd(v0, v1), v2);
        _mm256_storeu_pd(out + i0, _mm256_mul_pd(L, R));
      }
      for (int64_t i = i0; i < hi; ++i)
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
#else
    for (int64_t i = lo; i < hi; ++i)
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
#endif
  }
}
