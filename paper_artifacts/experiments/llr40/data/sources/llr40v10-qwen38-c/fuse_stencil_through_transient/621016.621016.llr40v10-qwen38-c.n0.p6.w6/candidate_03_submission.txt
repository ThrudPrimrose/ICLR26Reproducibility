#include <stdint.h>
#include <immintrin.h>

#define BLK 4096

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  int64_t n = LEN_1D - 3;
  if (n < 1) return;
  int64_t nblk = (n + BLK - 1) / BLK;
#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < nblk; ++b) {
    int64_t s = b * BLK + 1, e = s + BLK;
    if (e > n + 1) e = n + 1;
    int64_t j = s;
#if defined(__AVX512F__)
    if (__builtin_cpu_supports("avx512f")) {
      const double *pa = a - 1;
      for (; j + 7 < e; j += 8) {
        const double *p = pa + j;
        __m512d v0 = _mm512_loadu_pd(p);
        __m512d v1 = _mm512_loadu_pd(p + 1);
        __m512d v2 = _mm512_loadu_pd(p + 2);
        __m512d v3 = _mm512_loadu_pd(p + 3);
        __m512d l = _mm512_add_pd(_mm512_add_pd(v0, v1), v2);
        __m512d r = _mm512_add_pd(_mm512_add_pd(v1, v2), v3);
        _mm512_storeu_pd(out + j, _mm512_mul_pd(l, r));
      }
    }
#endif
    for (; j < e; ++j)
      out[j] = (a[j - 1] + a[j] + a[j + 1]) * (a[j] + a[j + 1] + a[j + 2]);
  }
}
