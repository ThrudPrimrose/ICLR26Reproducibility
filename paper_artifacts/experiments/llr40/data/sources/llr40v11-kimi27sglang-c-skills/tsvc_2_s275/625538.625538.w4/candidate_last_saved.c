#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  if (LEN_2D <= 0) return;
  const int64_t n = LEN_2D;
  const __m256d vzero = _mm256_setzero_pd();

  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i += 4) {
    const int64_t rem = n - i;
    if (rem >= 4) {
      __m256d a0 = _mm256_loadu_pd(&aa[i]);
      __m256d cond = _mm256_cmp_pd(a0, vzero, _CMP_GT_OQ);
      __m256i mask = _mm256_castpd_si256(cond);
      if (!_mm256_testz_si256(mask, mask)) {
        __m256d running = a0;
        for (int64_t j = 1; j < n; j++) {
          __m256d b = _mm256_loadu_pd(&bb[j * n + i]);
          __m256d c = _mm256_loadu_pd(&cc[j * n + i]);
          running = _mm256_fmadd_pd(b, c, running);
          _mm256_maskstore_pd(&aa[j * n + i], mask, running);
        }
      }
    } else {
      for (int64_t k = 0; k < rem; k++) {
        int64_t col = i + k;
        if (aa[col] > 0.0) {
          double running = aa[col];
          for (int64_t j = 1; j < n; j++) {
            running += bb[j * n + col] * cc[j * n + col];
            aa[j * n + col] = running;
          }
        }
      }
    }
  }
}
