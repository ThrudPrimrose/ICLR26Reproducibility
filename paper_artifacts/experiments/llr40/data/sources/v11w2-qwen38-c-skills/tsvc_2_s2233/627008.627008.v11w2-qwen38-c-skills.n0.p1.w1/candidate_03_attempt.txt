#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t Wtot = N - 8;               /* columns 8 .. N-1 */
  const int64_t base8 = (Wtot / 8) * 8;     /* width with full 8-cols per unit */
  const int nt = omp_get_max_threads();
  const int aligned = (((uintptr_t)aa | (uintptr_t)bb | (uintptr_t)cc) & 63) == 0;

  if (aligned) {
    #pragma omp parallel
    {
      const int t = omp_get_thread_num();
      const int64_t per = (base8 + nt - 1) / nt;
      const int64_t C0 = 8 + t * per;
      int64_t W = (C0 >= 8 + base8) ? 0 : (8 + base8 - C0);
      if (W > per) W = per;
      if (W > 0) {
        double *va = (double *)__builtin_alloca((size_t)W * sizeof(double));
        double *vb = (double *)__builtin_alloca((size_t)W * sizeof(double));
        for (int64_t k = 0; k < W; ++k) { va[k] = aa[7 * N + C0 + k]; vb[k] = bb[7 * N + C0 + k]; }
        int64_t p = (8 * N + C0) & 7;
        const int64_t nstep = N & 7;
        for (int64_t j = 8; j < N; ++j) {
          const int64_t delta = (8 - p) & 7;
          const int64_t row = j * N;
          const int64_t lead = delta < W ? delta : W;
          int64_t s = 0;
          for (; s < lead; ++s) {
            const int64_t idx = row + C0 + s;
            const double x = cc[idx];
            va[s] += x; aa[idx] = va[s];
            vb[s] += x; bb[idx] = vb[s];
          }
          const int64_t Q = (W - lead) >> 3;
          const double *ccw = cc + row + C0 + lead;
          double *aaw = aa + row + C0 + lead;
          double *bbw = bb + row + C0 + lead;
          for (int64_t q = 0; q < Q; ++q) {
            __m512d x = _mm512_load_pd(ccw + 8 * q);
            __m512d a = _mm512_loadu_pd(va + lead + 8 * q);
            __m512d b = _mm512_loadu_pd(vb + lead + 8 * q);
            a = _mm512_add_pd(a, x);
            b = _mm512_add_pd(b, x);
            _mm512_store_pd(aaw + 8 * q, a);
            _mm512_store_pd(bbw + 8 * q, b);
            _mm512_storeu_pd(va + lead + 8 * q, a);
            _mm512_storeu_pd(vb + lead + 8 * q, b);
          }
          s += 8 * Q;
          for (; s < W; ++s) {
            const int64_t idx = row + C0 + s;
            const double x = cc[idx];
            va[s] += x; aa[idx] = va[s];
            vb[s] += x; bb[idx] = vb[s];
          }
          p = (p + nstep) & 7;
        }
      }
      for (int64_t c = 8 + base8 + t; c < N; c += nt) {
        double va = aa[7 * N + c];
        double vb = bb[7 * N + c];
        for (int64_t j = 8; j < N; ++j) {
          const double x = cc[j * N + c];
          va += x; aa[j * N + c] = va;
          vb += x; bb[j * N + c] = vb;
        }
      }
    }
  } else {
    const int64_t last8 = (N - 8) & ~7;
    #pragma omp parallel for schedule(static)
    for (int64_t c = 8; c <= last8; c += 8) {
      __m512d va = _mm512_loadu_pd(aa + 7 * N + c);
      __m512d vb = _mm512_loadu_pd(bb + 7 * N + c);
      for (int64_t j = 8; j < N; ++j) {
        __m512d x = _mm512_loadu_pd(cc + j * N + c);
        va = _mm512_add_pd(va, x);
        vb = _mm512_add_pd(vb, x);
        _mm512_storeu_pd(aa + j * N + c, va);
        _mm512_storeu_pd(bb + j * N + c, vb);
      }
    }
    for (int64_t c = (last8 >= 8 ? last8 + 8 : 8); c < N; ++c) {
      double va = aa[7 * N + c];
      double vb = bb[7 * N + c];
      for (int64_t j = 8; j < N; ++j) {
        const double x = cc[j * N + c];
        va += x; aa[j * N + c] = va;
        vb += x; bb[j * N + c] = vb;
      }
    }
  }
}
