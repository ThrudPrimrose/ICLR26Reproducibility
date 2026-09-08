#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t ncol = N - 8;
  const int64_t nfull = (ncol / 8) * 8;
  const int64_t ngroups = nfull / 8;

  #pragma omp parallel for schedule(static)
  for (int64_t g = 0; g < ngroups; ++g) {
    const int64_t k0 = 8 + g*8;
    __m512d vA = _mm512_loadu_pd(aa + 7*N + k0);
    __m512d vB = _mm512_loadu_pd(bb + 7*N + k0);
    for (int64_t j = 8; j < N; ++j) {
      __m512d c = _mm512_loadu_pd(cc + j*N + k0);
      vA = _mm512_add_pd(vA, c);
      vB = _mm512_add_pd(vB, c);
      _mm512_storeu_pd(aa + j*N + k0, vA);
      _mm512_storeu_pd(bb + j*N + k0, vB);
    }
  }

  /* tail columns (scalar) */
  for (int64_t k = 8 + nfull; k < N; ++k) {
    double vA = aa[7*N+k], vB = bb[7*N+k];
    for (int64_t j = 8; j < N; ++j) {
      double c = cc[j*N+k];
      vA += c; aa[j*N+k] = vA;
      vB += c; bb[j*N+k] = vB;
    }
  }
}
