#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC s275: for each column i, if aa[0,i] > 0 then
   aa[j,i] = aa[j-1,i] + bb[j,i]*cc[j,i], j=1..L-1 (serial scan per column).
   Columns are independent: thread over blocks of 8 consecutive columns.
   8 consecutive columns at the same row j are 64 contiguous bytes, so each
   cache line is fully utilized (no strided 8x DRAM amplification) and the
   8 serial scans run as one AVX-512 masked-FMA chain. The guard aa[0,i]>0
   is a per-lane mask, fixed for the whole block. */
#if defined(__AVX512F__)
static void __attribute__((noinline))
block_scan(double *restrict aa, const double *restrict bb, const double *restrict cc,
           const int64_t L, const int64_t i0, const int64_t rem) {
  __mmask8 valid = 0u;
  for (int64_t k = 0; k < rem; k++) valid |= (1u << (unsigned)k);
  __m512d s = _mm512_maskz_loadu_pd(valid, aa + i0);
  const __mmask8 act = _mm512_cmp_pd_mask(s, _mm512_setzero_pd(), _CMP_GT_OQ) & valid;
  if (act == 0) return;
  for (int64_t j = 1; j < L; j++) {
    const int64_t base = j * L + i0;
    const __m512d b = _mm512_maskz_loadu_pd(act, bb + base);
    const __m512d c = _mm512_maskz_loadu_pd(act, cc + base);
    s = _mm512_mask_fmadd_pd(b, act, c, s);  /* s + b*c ; inactive lanes keep s */
    _mm512_mask_storeu_pd(aa + base, act, s);
  }
}
#endif

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  if (L <= 1) return;
  const int64_t nblk = (L + 7) / 8;

#if defined(__AVX512F__)
  #pragma omp parallel for schedule(static)
  for (int64_t bl = 0; bl < nblk; bl++)
    block_scan(aa, bb, cc, L, bl * 8, L - bl * 8);
#else
  #pragma omp parallel for schedule(static)
  for (int64_t bl = 0; bl < nblk; bl++) {
    const int64_t i0 = bl * 8;
    const int64_t rem = L - i0;
    char act[8];
    double s[8];
    for (int64_t k = 0; k < 8; k++) {
      if (k < rem && aa[i0 + k] > 0.0) { act[k] = 1; s[k] = aa[i0 + k]; }
      else act[k] = 0;
    }
    for (int64_t j = 1; j < L; j++) {
      const int64_t base = j * L + i0;
      for (int64_t k = 0; k < rem; k++) {
        if (act[k]) {
          s[k] += bb[base + k] * cc[base + k];
          aa[base + k] = s[k];
        }
      }
    }
  }
#endif
}
