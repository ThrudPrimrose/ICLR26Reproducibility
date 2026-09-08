#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* tsvc_2_s4112:  a[i] += b[ip[i]] * 2.0
 * Fully parallel elementwise loop; the only cost is the random gather of b.
 * Per thread: 4 independent AVX-512 8-wide gather chains keep the load
 * queue full; scalar fallback for small N to avoid fork overhead. */
void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  if (LEN_1D < (1LL << 19)) {
    for (int64_t i = 0; i < LEN_1D; ++i) {
      a[i] += b[ip[i]] * 2.0;
    }
    return;
  }

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t base = 0; base < LEN_1D; base += 32) {
      int64_t n = LEN_1D - base;
      if (n >= 32) {
        __m256i i0 = _mm256_loadu_si256((const __m256i_u *)(ip + base));
        __m256i i1 = _mm256_loadu_si256((const __m256i_u *)(ip + base + 8));
        __m256i i2 = _mm256_loadu_si256((const __m256i_u *)(ip + base + 16));
        __m256i i3 = _mm256_loadu_si256((const __m256i_u *)(ip + base + 24));
        __m512d v0 = _mm512_i32gather_pd(i0, b, 8);
        __m512d v1 = _mm512_i32gather_pd(i1, b, 8);
        __m512d v2 = _mm512_i32gather_pd(i2, b, 8);
        __m512d v3 = _mm512_i32gather_pd(i3, b, 8);
        __m512d a0 = _mm512_loadu_pd(a + base);
        __m512d a1 = _mm512_loadu_pd(a + base + 8);
        __m512d a2 = _mm512_loadu_pd(a + base + 16);
        __m512d a3 = _mm512_loadu_pd(a + base + 24);
        _mm512_storeu_pd(a + base,     _mm512_add_pd(a0, _mm512_mul_pd(v0, _mm512_set1_pd(2.0))));
        _mm512_storeu_pd(a + base + 8, _mm512_add_pd(a1, _mm512_mul_pd(v1, _mm512_set1_pd(2.0))));
        _mm512_storeu_pd(a + base + 16,_mm512_add_pd(a2, _mm512_mul_pd(v2, _mm512_set1_pd(2.0))));
        _mm512_storeu_pd(a + base + 24,_mm512_add_pd(a3, _mm512_mul_pd(v3, _mm512_set1_pd(2.0))));
      } else {
        for (int64_t i = base; i < LEN_1D; ++i) {
          a[i] += b[ip[i]] * 2.0;
        }
      }
    }
  }
}
