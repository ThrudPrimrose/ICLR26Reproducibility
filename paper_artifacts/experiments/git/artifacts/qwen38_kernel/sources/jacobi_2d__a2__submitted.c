#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static inline void sweep_row_generic(const double *restrict U,
                                     const double *restrict C,
                                     const double *restrict D,
                                     double *restrict O, int64_t N) {
  for (int64_t j = 1; j < N - 1; ++j)
    O[j] = 0.2 * ((C[j] + C[j - 1]) + C[j + 1] + U[j] + D[j]);
}

#if defined(__AVX512F__)
static inline int64_t bits64(double x) {
  int64_t r;
  __builtin_memcpy(&r, &x, 8);
  return r;
}

static inline void sweep_row_avx512(const double *restrict U,
                                    const double *restrict C,
                                    const double *restrict D,
                                    double *restrict O, int64_t N) {
  if (N < 18) { sweep_row_generic(U, C, D, O, N); return; }
  const __m512d vhalf = _mm512_set1_pd(0.2);
  const __m512i cntL = _mm512_mask_set1_epi64(_mm512_setzero_si512(), 0x1, 8);
  const __m512i cntR = _mm512_mask_set1_epi64(_mm512_setzero_si512(), 0x80, 8);
  int64_t b = 1;
  double prev = C[0];
  for (; b + 16 <= N - 1; b += 16) {
    __m512d c = _mm512_loadu_pd(C + b);
    double right = C[b + 16];
    __m512i ci  = _mm512_castpd_si512(c);
    __m512i cl  = _mm512_sllv_epi64(ci, cntL);
    cl = _mm512_mask_set1_epi64(cl, 0x1, bits64(prev));
    __m512i cr  = _mm512_srlv_epi64(ci, cntR);
    cr = _mm512_mask_set1_epi64(cr, 0x80, bits64(right));
    __m512d u = _mm512_loadu_pd(U + b);
    __m512d d = _mm512_loadu_pd(D + b);
    __m512d s = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(c, _mm512_castsi512_pd(cl)),
                                            _mm512_castsi512_pd(cr)), u);
    s = _mm512_mul_pd(_mm512_add_pd(s, d), vhalf);
    _mm512_storeu_pd(O + b, s);
    prev = right;
  }
  for (int64_t j = b; j < N - 1; ++j)
    O[j] = 0.2 * ((C[j] + C[j - 1]) + C[j + 1] + U[j] + D[j]);
}
#endif

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
  for (int64_t t = 0; t < TSTEPS; ++t) {
    #pragma omp parallel for schedule(static, 4)
    for (int64_t i = 1; i < N - 1; ++i) {
      const double *restrict U = A + (i - 1) * N;
      const double *restrict C = A + i * N;
      const double *restrict D = A + (i + 1) * N;
      double *restrict O = B + i * N;
#if defined(__AVX512F__)
      sweep_row_avx512(U, C, D, O, N);
#else
      sweep_row_generic(U, C, D, O, N);
#endif
    }
    #pragma omp parallel for schedule(static, 4)
    for (int64_t i = 1; i < N - 1; ++i) {
      const double *restrict U = B + (i - 1) * N;
      const double *restrict C = B + i * N;
      const double *restrict D = B + (i + 1) * N;
      double *restrict O = A + i * N;
#if defined(__AVX512F__)
      sweep_row_avx512(U, C, D, O, N);
#else
      sweep_row_generic(U, C, D, O, N);
#endif
    }
  }
}
