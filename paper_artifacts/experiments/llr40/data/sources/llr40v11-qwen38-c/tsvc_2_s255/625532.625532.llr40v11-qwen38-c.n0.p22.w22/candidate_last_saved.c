#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D < 2) return;
  a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
  if (LEN_1D < 3) return;
  a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;

  const int64_t n = LEN_1D;
  const int64_t nvec = 2 + ((n - 2) & ~7); /* multiple-of-4 start at 2, keep 32B-friendly stride */
  const __m256d c = _mm256_set1_pd(0.333);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 2; i < nvec; i += 4) {
    __m256d v0 = _mm256_loadu_pd(&b[i]);
    __m256d v1 = _mm256_loadu_pd(&b[i - 1]);
    __m256d v2 = _mm256_loadu_pd(&b[i - 2]);
    _mm256_storeu_pd(&a[i], _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v0, v1), v2), c));
  }
  for (int64_t i = nvec; i < n; i++)
    a[i] = ((b[i] + b[i - 1]) + b[i - 2]) * 0.333;
}
