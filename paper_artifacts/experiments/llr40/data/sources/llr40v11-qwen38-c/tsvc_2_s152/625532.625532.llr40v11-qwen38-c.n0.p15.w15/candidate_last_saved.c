#include <stdint.h>
#include <immintrin.h>

/* b[i] = d[i]*e[i]; a[i] += b[i]*c[i]
 * AVX-512, 64-element chunks unrolled x8 (64 doubles); tail matches the
 * reference contraction (FMA under -ffp-contract=fast). Unaligned intrinsics. */
void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  #pragma omp parallel for schedule(static)
  for (int64_t base = 0; base < LEN_1D; base += 128) {
    int64_t n = LEN_1D - base;
    if (n >= 128) {
      const double *dp = d + base;
      const double *ep = e + base;
      const double *cp = c + base;
      double *ap = a + base;
      double *bp = b + base;
      #pragma GCC unroll 16
      for (int k = 0; k < 16; ++k) {
        __m512d vd = _mm512_loadu_pd(dp + 8 * k);
        __m512d ve = _mm512_loadu_pd(ep + 8 * k);
        __m512d vb = _mm512_mul_pd(vd, ve);
        _mm512_stream_pd(bp + 8 * k, vb);
        __m512d va = _mm512_loadu_pd(ap + 8 * k);
        __m512d vc = _mm512_loadu_pd(cp + 8 * k);
        va = _mm512_fmadd_pd(vb, vc, va);
        _mm512_storeu_pd(ap + 8 * k, va);
      }
    } else {
      for (int64_t i = base; i < LEN_1D; ++i) {
        const double t = d[i] * e[i];
        b[i] = t;
        a[i] += t * c[i];
      }
    }
  }
}
