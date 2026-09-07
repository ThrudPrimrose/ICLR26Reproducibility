#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#if defined(__AVX512F__)
static inline void sweep_row(const double *restrict r0, const double *restrict r1,
                             const double *restrict r2, double *restrict d, int64_t N) {
  int64_t j = 1;
  int64_t jv_end = N - 9;
  for (; j <= jv_end; j += 8) {
    __m512d s = _mm512_loadu_pd(r1 + j);
    s = _mm512_add_pd(s, _mm512_loadu_pd(r1 + j - 1));
    s = _mm512_add_pd(s, _mm512_loadu_pd(r1 + j + 1));
    s = _mm512_add_pd(s, _mm512_loadu_pd(r2 + j));
    s = _mm512_add_pd(s, _mm512_loadu_pd(r0 + j));
    _mm512_storeu_pd(d + j, _mm512_mul_pd(s, _mm512_set1_pd(0.2)));
  }
  for (; j < N - 1; ++j)
    d[j] = 0.2 * (r1[j] + r1[j - 1] + r1[j + 1] + r2[j] + r0[j]);
}
#else
static inline void sweep_row(const double *restrict r0, const double *restrict r1,
                             const double *restrict r2, double *restrict d, int64_t N) {
  int64_t j = 1;
  int64_t jv_end = N - 5;
  for (; j <= jv_end; j += 4) {
    __m256d s = _mm256_loadu_pd(r1 + j);
    s = _mm256_add_pd(s, _mm256_loadu_pd(r1 + j - 1));
    s = _mm256_add_pd(s, _mm256_loadu_pd(r1 + j + 1));
    s = _mm256_add_pd(s, _mm256_loadu_pd(r2 + j));
    s = _mm256_add_pd(s, _mm256_loadu_pd(r0 + j));
    _mm256_storeu_pd(d + j, _mm256_mul_pd(s, _mm256_set1_pd(0.2)));
  }
  for (; j < N - 1; ++j)
    d[j] = 0.2 * (r1[j] + r1[j - 1] + r1[j + 1] + r2[j] + r0[j]);
}
#endif

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
  for (int64_t t = 0; t < TSTEPS; ++t) {
    #pragma omp parallel
    {
      #pragma omp for schedule(static)
      for (int64_t i = 1; i < N - 1; ++i)
        sweep_row(A + (i - 1) * N, A + i * N, A + (i + 1) * N, B + i * N, N);
      #pragma omp for schedule(static)
      for (int64_t i = 1; i < N - 1; ++i)
        sweep_row(B + (i - 1) * N, B + i * N, B + (i + 1) * N, A + i * N, N);
    }
  }
}
