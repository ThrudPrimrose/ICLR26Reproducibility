/* TSVC tsvc_2 "vag": a[i] = b[ip[i]] -- vectorized gather + OpenMP. */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void gather_block_avx512(double *a, const double *b, const int32_t *ip,
                                       int64_t i0, int64_t i1) {
  for (; i0 + 8 <= i1; i0 += 8) {
    __m256i vix = _mm256_loadu_si256((const __m256i *)(ip + i0));
    _mm512_storeu_pd(a + i0, _mm512_i32gather_pd(vix, b, 8));
  }
  for (; i0 < i1; ++i0) a[i0] = b[ip[i0]];
}

static inline void gather_block_avx2(double *a, const double *b, const int32_t *ip,
                                     int64_t i0, int64_t i1) {
  for (; i0 + 4 <= i1; i0 += 4) {
    __m128i vix = _mm_loadu_si128((const __m128i *)(ip + i0));
    _mm256_storeu_pd(a + i0, _mm256_i32gather_pd(b, vix, 8));
  }
  for (; i0 < i1; ++i0) a[i0] = b[ip[i0]];
}

static inline void gather_block_scalar(double *a, const double *b, const int32_t *ip,
                                       int64_t i0, int64_t i1) {
  for (; i0 < i1; ++i0) a[i0] = b[ip[i0]];
}

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b,
                     const int32_t *restrict ip, const int64_t LEN_1D) {
#if defined(__AVX512F__)
  if (__builtin_cpu_supports("avx512f")) {
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < LEN_1D; i0 += 8) {
      int64_t i1 = i0 + 8;
      if (i1 > LEN_1D) i1 = LEN_1D;
      gather_block_avx512(a, b, ip, i0, i1);
    }
    return;
  }
#endif
#if defined(__AVX2__)
  if (__builtin_cpu_supports("avx2")) {
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < LEN_1D; i0 += 4) {
      int64_t i1 = i0 + 4;
      if (i1 > LEN_1D) i1 = LEN_1D;
      gather_block_avx2(a, b, ip, i0, i1);
    }
    return;
  }
#endif
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) a[i] = b[ip[i]];
}
