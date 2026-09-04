#include <stdint.h>
#include <omp.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  #pragma omp parallel for schedule(static)
  for (int64_t k = 8; k < N; k += 8) {
    const int64_t cnt = (N - k < 8) ? (N - k) : 8;
    if (cnt == 8) {
      {
        __m128d s0 = _mm_loadu_pd(aa + 7 * N + k);
        __m128d s1 = _mm_loadu_pd(aa + 7 * N + k + 2);
        __m128d s2 = _mm_loadu_pd(aa + 7 * N + k + 4);
        __m128d s3 = _mm_loadu_pd(aa + 7 * N + k + 6);
        for (int64_t j = 8; j < N; ++j) {
          const double *c = cc + j * N + k;
          double *a = aa + j * N + k;
          s0 = _mm_add_pd(s0, _mm_loadu_pd(c + 0));
          s1 = _mm_add_pd(s1, _mm_loadu_pd(c + 2));
          s2 = _mm_add_pd(s2, _mm_loadu_pd(c + 4));
          s3 = _mm_add_pd(s3, _mm_loadu_pd(c + 6));
          _mm_storeu_pd(a + 0, s0);
          _mm_storeu_pd(a + 2, s1);
          _mm_storeu_pd(a + 4, s2);
          _mm_storeu_pd(a + 6, s3);
        }
      }
      {
        __m128d s0 = _mm_loadu_pd(bb + 7 * N + k);
        __m128d s1 = _mm_loadu_pd(bb + 7 * N + k + 2);
        __m128d s2 = _mm_loadu_pd(bb + 7 * N + k + 4);
        __m128d s3 = _mm_loadu_pd(bb + 7 * N + k + 6);
        for (int64_t i = 8; i < N; ++i) {
          const double *c = cc + i * N + k;
          double *b = bb + i * N + k;
          s0 = _mm_add_pd(s0, _mm_loadu_pd(c + 0));
          s1 = _mm_add_pd(s1, _mm_loadu_pd(c + 2));
          s2 = _mm_add_pd(s2, _mm_loadu_pd(c + 4));
          s3 = _mm_add_pd(s3, _mm_loadu_pd(c + 6));
          _mm_storeu_pd(b + 0, s0);
          _mm_storeu_pd(b + 2, s1);
          _mm_storeu_pd(b + 4, s2);
          _mm_storeu_pd(b + 6, s3);
        }
      }
    } else {
      for (int g = 0; g < cnt; ++g) {
        double s = aa[7 * N + k + g];
        for (int64_t j = 8; j < N; ++j) { s += cc[j * N + k + g]; aa[j * N + k + g] = s; }
        s = bb[7 * N + k + g];
        for (int64_t i = 8; i < N; ++i) { s += cc[i * N + k + g]; bb[i * N + k + g] = s; }
      }
    }
  }
}
#else
void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  for (int64_t i = 8; i < LEN_2D; ++i) {
    double s = aa[7 * LEN_2D + i];
    for (int64_t j = 8; j < LEN_2D; ++j) { s += cc[j * LEN_2D + i]; aa[j * LEN_2D + i] = s; }
  }
  for (int64_t j = 8; j < LEN_2D; ++j) {
    double s = bb[7 * LEN_2D + j];
    for (int64_t i = 8; i < LEN_2D; ++i) { s += cc[i * LEN_2D + j]; bb[i * LEN_2D + j] = s; }
  }
}
#endif
