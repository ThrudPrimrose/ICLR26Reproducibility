#include <stdint.h>
#include <immintrin.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  const int64_t iend = LEN_1D - 3;
  if (iend < 16) {
    for (int64_t i = 1; i <= iend; ++i)
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    return;
  }
  for (int64_t i = 1; i < 16; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
  const int64_t iv1 = (iend >= 23) ? (iend & ~15LL) : 16;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 16; i < iv1; i += 16) {
    const double *p0 = a + i - 1;
    const double *p1 = a + i + 7;
    __m512d v0 = _mm512_loadu_pd(p0);
    __m512d v1 = _mm512_loadu_pd(p0 + 1);
    __m512d v2 = _mm512_loadu_pd(p0 + 2);
    __m512d v3 = _mm512_loadu_pd(p0 + 3);
    _mm512_store_pd(out + i,
                    _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v0, v1), v2),
                                  _mm512_add_pd(_mm512_add_pd(v1, v2), v3)));
    __m512d w0 = _mm512_loadu_pd(p1);
    __m512d w1 = _mm512_loadu_pd(p1 + 1);
    __m512d w2 = _mm512_loadu_pd(p1 + 2);
    __m512d w3 = _mm512_loadu_pd(p1 + 3);
    _mm512_store_pd(out + i + 8,
                    _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(w0, w1), w2),
                                  _mm512_add_pd(_mm512_add_pd(w1, w2), w3)));
  }
  for (int64_t i = iv1; i <= iend; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
