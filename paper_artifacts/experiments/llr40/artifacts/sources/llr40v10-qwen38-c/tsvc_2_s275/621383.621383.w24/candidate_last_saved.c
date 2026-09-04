#include <stdint.h>
#include <immintrin.h>

/* v4: S-stage delay line + deep _mm_prefetch ahead; K batches interleaved per thread */
#define STAGES 2
#define PB 16
#define K 4

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  const int64_t nbatch = (L + 7) / 8;
  const int64_t niter = (nbatch + K - 1) / K;

  #pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < niter; t++) {
    int64_t i0[K] = {0};
    double *a[K] = {0};
    const double *b[K] = {0};
    const double *c[K] = {0};
    __m512d acc[K] = {0};
    __mmask8 m[K] = {0};
    __m512d vb[K][STAGES] = {0}, vc[K][STAGES] = {0};
    int live[K] = {0};

    int64_t nvalid = nbatch - (int64_t)t * K;
    if (nvalid < 1) continue;
    if (nvalid < K) nvalid = (int64_t)K;

    for (int k = 0; k < K; k++) {
      const int64_t i0k = (int64_t)t * K + k;
      i0[k] = (i0k < nbatch) ? i0k * 8 : 0;
      a[k] = aa + i0[k];
      b[k] = bb + i0[k];
      c[k] = cc + i0[k];
      if (i0k >= nbatch || L - i0[k] < 8) { live[k] = 0; m[k] = 0; continue; }
      __m512d v0 = _mm512_loadu_pd(a[k]);
      m[k] = _mm512_cmp_pd_mask(v0, _mm512_setzero_pd(), _CMP_GT_OQ);
      if (m[k] == 0) { live[k] = 0; continue; }
      live[k] = 1;
      acc[k] = v0;
      for (int s = 0; s < STAGES; s++) {
        int64_t r = 1 + s;
        if (r < L) { vb[k][s] = _mm512_loadu_pd(b[k] + r * L); vc[k][s] = _mm512_loadu_pd(c[k] + r * L); }
        else { vb[k][s] = _mm512_setzero_pd(); vc[k][s] = _mm512_setzero_pd(); }
      }
    }

    int64_t jend = L - STAGES; if (jend < 1) jend = 1;
    for (int64_t j = 1; j < L; j++) {
      for (int k = 0; k < K; k++) {
        if (!live[k]) continue;
        __m512d nv, nc;
        if (j < jend) {
          nv = _mm512_loadu_pd(b[k] + (j + STAGES) * L);
          nc = _mm512_loadu_pd(c[k] + (j + STAGES) * L);
          if (j + PB < L) { _mm_prefetch((const char *)(b[k] + (j + PB) * L), _MM_HINT_T0); _mm_prefetch((const char *)(c[k] + (j + PB) * L), _MM_HINT_T0); }
        } else {
          nv = _mm512_setzero_pd(); nc = _mm512_setzero_pd();
          if (j + STAGES < L) { nv = _mm512_loadu_pd(b[k] + (j + STAGES) * L); nc = _mm512_loadu_pd(c[k] + (j + STAGES) * L); }
        }
        acc[k] = _mm512_fmadd_pd(vb[k][0], vc[k][0], acc[k]);
        _mm512_mask_storeu_pd(a[k] + j * L, m[k], acc[k]);
        for (int s = 0; s < STAGES - 1; s++) { vb[k][s] = vb[k][s + 1]; vc[k][s] = vc[k][s + 1]; }
        vb[k][STAGES - 1] = nv; vc[k][STAGES - 1] = nc;
      }
    }

    /* scalar tail for partial final batch (cnt < 8) */
    if (t == niter - 1) {
      int64_t i0t = (nbatch - 1) * 8;
      int64_t cnt = L - i0t;
      if (cnt < 8) {
        const double * __restrict b2 = bb + i0t;
        const double * __restrict c2 = cc + i0t;
        double * __restrict a2 = aa + i0t;
        for (int64_t k = 0; k < cnt; k++)
          if (a2[k] > 0.0)
            for (int64_t j = 1; j < L; j++)
              a2[j * L + k] = a2[(j - 1) * L + k] + b2[j * L + k] * c2[j * L + k];
      }
    }
  }
}
