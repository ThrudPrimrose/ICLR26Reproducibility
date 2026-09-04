#include <stdint.h>
#include <immintrin.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  const int64_t nvec = LEN_1D & ~63;
  #pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < nvec; j += 64) {
    __m512d va[8], vb[8], vc[8];
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) va[k] = _mm512_load_pd(a + j + 8*k);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) vb[k] = _mm512_load_pd(b + j + 8*k);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) vc[k] = _mm512_load_pd(c + j + 8*k);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k)
      _mm512_store_pd(a + j + 8*k, _mm512_mul_pd(va[k], _mm512_mul_pd(vb[k], vc[k])));
  }
  #pragma omp parallel for schedule(static)
  for (int64_t i = nvec; i < LEN_1D; ++i) a[i] = a[i] * b[i] * c[i];
}
