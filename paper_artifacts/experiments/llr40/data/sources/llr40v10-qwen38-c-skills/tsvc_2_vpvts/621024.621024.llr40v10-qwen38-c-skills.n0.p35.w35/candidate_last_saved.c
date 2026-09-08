#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  const double sd = (double)S;
  const int64_t HALF = LEN_1D / 2;
  const int64_t nfull = HALF / 32;
  #pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < nfull; ++c) {
    const int64_t j = c * 32;
    const int64_t k = j + HALF;
    __m512d s = _mm512_set1_pd(sd);
    for (int64_t off = 0; off < 32; off += 16) {
      __m512d a0 = _mm512_loadu_pd(a + j + off),     b0 = _mm512_loadu_pd(b + j + off);
      __m512d a1 = _mm512_loadu_pd(a + j + off + 8), b1 = _mm512_loadu_pd(b + j + off + 8);
      __m512d a2 = _mm512_loadu_pd(a + k + off),     b2 = _mm512_loadu_pd(b + k + off);
      __m512d a3 = _mm512_loadu_pd(a + k + off + 8), b3 = _mm512_loadu_pd(b + k + off + 8);
      _mm512_storeu_pd(a + j + off,     _mm512_add_pd(a0, _mm512_mul_pd(b0, s)));
      _mm512_storeu_pd(a + j + off + 8, _mm512_add_pd(a1, _mm512_mul_pd(b1, s)));
      _mm512_storeu_pd(a + k + off,     _mm512_add_pd(a2, _mm512_mul_pd(b2, s)));
      _mm512_storeu_pd(a + k + off + 8, _mm512_add_pd(a3, _mm512_mul_pd(b3, s)));
    }
  }
  for (int64_t i = nfull * 32; i < HALF; ++i) a[i] += b[i] * sd;
  for (int64_t i = HALF + nfull * 32; i < LEN_1D; ++i) a[i] += b[i] * sd;
}
