#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  const int64_t n = L - 8;
  if (n <= 0) return;

  // Part A: aa[j*L+i] = aa[7*L+i] + sum_{k=8..j} cc[k*L+i]; independent per column,
  // one 512-bit running sum across 8 columns.
  #pragma omp parallel for schedule(static)
  for (int64_t i0 = 0; i0 < n; i0 += 8) {
    const int64_t cnt = (n - i0 < 8) ? (n - i0) : 8;
    double *a0 = aa + 7 * L + 8 + i0;
    if (cnt < 8) {
      double run[8];
      for (int64_t t = 0; t < cnt; ++t) run[t] = a0[t];
      for (int64_t j = 8; j < L; ++j) {
        const double *c = cc + j * L + 8 + i0;
        double *o = a0 + (j - 7) * L;
        for (int64_t t = 0; t < cnt; ++t) { run[t] += c[t]; o[t] = run[t]; }
      }
    } else {
      __m512d run = _mm512_loadu_pd(a0);
      for (int64_t j = 8; j < L; ++j) {
        run = _mm512_add_pd(run, _mm512_loadu_pd(cc + j * L + 8 + i0));
        _mm512_storeu_pd(a0 + (j - 7) * L, run);
      }
    }
  }

  // Part B: bb[j*L+i] = bb[j*L+7] + scan of cc row j from i=8; independent per row,
  // SIMD tree prefix (width 8) with carry, scalar head/tail.
  #pragma omp parallel for schedule(static)
  for (int64_t j = 8; j < L; ++j) {
    const double *c = cc + j * L + 8;
    double *o = bb + j * L + 8;
    double carry = bb[j * L + 7];
    if (n < 8) {
      for (int64_t k = 0; k < n; ++k) { carry += c[k]; o[k] = carry; }
    } else {
      int64_t k = 0;
      double x[16];
      const int64_t xl = (n < 16) ? n : 16;
      for (int64_t t = 0; t < xl; ++t) x[t] = c[t];
      for (int64_t t = 0; t < 8; ++t) { carry += x[t]; o[t] = carry; }
      __m512d p = _mm512_loadu_pd(x + 8);
      k = 8;
      for (; k + 8 <= n; k += 8) {
        __m512i vi = _mm512_castpd_si512(p);
        __m512d v1 = _mm512_castsi512_pd(_mm512_slli_epi32(vi, 32));
        __m512d v2 = _mm512_castsi512_pd(_mm512_slli_epi32(vi, 64));
        __m512d v4 = _mm512_castsi512_pd(_mm512_slli_epi32(v2, 64));
        p = _mm512_add_pd(p, v1);
        p = _mm512_add_pd(p, v2);
        p = _mm512_add_pd(p, v4);
        p = _mm512_add_pd(p, _mm512_set1_pd(carry));
        _mm512_storeu_pd(o + k, p);
        {
          double buf[8];
          _mm512_storeu_pd(buf, p);
          carry = buf[7];
        }
      }
      for (; k < n; ++k) { carry += c[k]; o[k] = carry; }
    }
  }
}
