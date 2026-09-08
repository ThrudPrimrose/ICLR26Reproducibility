#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;

  // Phase 1: a[i] = fma(b[i], c[i], a[i])  (reference contracts this to an FMA)
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) a[i] = fma(b[i], c[i], a[i]);

  // Phase 2: column-wise scans.  aa[j,i] = fma(bb[j,i], a[i], aa[j-1,i]), j=1..N-1.
  // Process 8 contiguous columns per 512-bit step; carry lives in registers.
  const int64_t nch = (N + 7) / 8;
  #pragma omp parallel for schedule(static)
  for (int64_t ch = 0; ch < nch; ++ch) {
    const int64_t i0 = ch * 8;
    const int64_t rem = N - i0;
    if (rem == 8) {
      __m512d carry = _mm512_loadu_pd(aa + i0);
      __m512d av = _mm512_loadu_pd(a + i0);
      const double *pbb = bb + N + i0;
      double *paa = aa + N + i0;
      for (int64_t j = 1; j < N; ++j) {
        __m512d v = _mm512_loadu_pd(pbb);
        carry = _mm512_fmadd_pd(v, av, carry);
        _mm512_storeu_pd(paa, carry);
        pbb += N;
        paa += N;
      }
    } else {
      for (int64_t k = 0; k < rem; ++k) {
        int64_t i = i0 + k;
        double carry = aa[i];
        for (int64_t j = 1; j < N; ++j) {
          carry = fma(bb[j * N + i], a[i], carry);
          aa[j * N + i] = carry;
        }
      }
    }
  }
}
