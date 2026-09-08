/* TSVC tsvc_2 s2233 fp64 -- optimized.
 *
 * Key structural fact (verified vs numpy reference):
 *   aa[j][i] = aa[j-1][i] + cc[j][i]   (j loop)  -> per column i, a running sum of cc down the
 *                                                  column, offset by aa[7][i]. The old aa[j][i]
 *                                                  (j>=8) is NEVER read.
 *   bb[i][j] = bb[i-1][j] + cc[i][j]   (i loop)  -> per column j, a running sum of cc down the
 *                                                  column, offset by bb[7][j]. The old bb[i][j]
 *                                                  (i>=8) is NEVER read.
 * Both use the SAME prefix sum of cc per column. So per column c:
 *   run=0; for s=8..n-1: run += cc[s][c]; aa[s][c] = aa[7][c] + run; bb[s][c] = bb[7][c] + run;
 * This reads cc once per column (contiguous 8-col groups via AVX-512) and writes aa,bb.
 * Columns are independent -> parallel over 8-column groups.
 */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 8) return;
  const int64_t ncol = n - 8;
  const int64_t ng = ncol / 8;
  const int64_t rem = ncol % 8;

  #pragma omp parallel for schedule(static)
  for (int64_t g = 0; g < ng; ++g) {
    const int64_t c0 = 8 + 8 * g;
    const double *ccp = cc + c0;
    double *aap = aa + c0;
    double *bbp = bb + c0;
    __m512d aoffv = _mm512_loadu_pd(aap + 7 * n);
    __m512d boffv = _mm512_loadu_pd(bbp + 7 * n);
    __m512d run = _mm512_setzero_pd();
    for (int64_t s = 8; s < n; ++s) {
      __m512d cv = _mm512_loadu_pd(ccp + s * n);
      run = _mm512_add_pd(run, cv);
      _mm512_storeu_pd(aap + s * n, _mm512_add_pd(aoffv, run));
      _mm512_storeu_pd(bbp + s * n, _mm512_add_pd(boffv, run));
    }
  }
  for (int64_t k = 0; k < rem; ++k) {
    const int64_t col = 8 + 8 * ng + k;
    const double aoff = aa[7 * n + col];
    const double boff = bb[7 * n + col];
    double run = 0.0;
    for (int64_t s = 8; s < n; ++s) {
      run += cc[s * n + col];
      aa[s * n + col] = aoff + run;
      bb[s * n + col] = boff + run;
    }
  }
}
