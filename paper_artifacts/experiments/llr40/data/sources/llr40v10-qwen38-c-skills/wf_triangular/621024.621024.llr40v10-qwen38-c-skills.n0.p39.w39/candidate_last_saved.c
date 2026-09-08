#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#ifndef PBLK
#define PBLK 64
#endif

/* wf_triangular: a[i][j] += a[i-1][j] + a[i][j-1] for 1<=i<L, i<=j<L.
 *
 * Dependence analysis: cell (i,j) reads only (i-1,j) and (i,j-1) -- both on
 * anti-diagonal t-1 where t = i+j. So the parallel front is the anti-diagonal.
 *
 * Blocked wavefront over P x P blocks: block (bi,bj) depends only on
 * (bi-1,bj) and (bi,bj-1), so all blocks with bi+bj = bd are independent and
 * are parallelized with `omp for`. Inside a block we scan rows (unit stride),
 * keeping the north/west edges in L1/L2; the off-chain add old+north is
 * 256-bit SIMD and the row scan is a 4-wide SIMD-scan per element group. */
void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D){
  const int64_t L = LEN_2D;
  const int64_t P = PBLK;
  const int64_t nb = (L + P - 1) / P;
  #pragma omp parallel
  {
    for (int64_t bd = 0; bd <= 2*nb-2; ++bd){
      int64_t bi_lo = bd - (nb-1); if (bi_lo < 0) bi_lo = 0;
      int64_t bi_hi = bd/2;        if (bi_hi > nb-1) bi_hi = nb-1;
      #pragma omp for schedule(static)
      for (int64_t bi = bi_lo; bi <= bi_hi; ++bi){
        const int64_t bj = bd - bi;
        const int64_t i1 = (bi+1)*P < L ? (bi+1)*P : L;
        int64_t i0 = bi*P; if (bi == 0) i0 = 1; /* row 0 is never updated */
        const int64_t j0 = bj*P;
        const int64_t j1 = (bj+1)*P < L ? (bj+1)*P : L;
        for (int64_t i = i0; i < i1; ++i){
          int64_t js = j0; if (js < i) js = i; /* triangle: j >= i */
          double *restrict r = a + i*L;
          const double *restrict n = a + (i-1)*L;
          const int64_t len = j1 - js;
          const int64_t n4 = len & ~3LL;
          double carry = r[js-1]; /* west seed; for js==i this is the never-updated a[i][i-1] */
          const double *nv = n + js;
          double *rv = r + js;
          for (int64_t k = 0; k < n4; k += 4){
            __m256d t = _mm256_add_pd(_mm256_loadu_pd(nv + k), _mm256_loadu_pd(rv + k));
            double u[4];
            _mm256_storeu_pd(u, t);
            u[0] += carry;
            u[1] += u[0];
            u[2] += u[1];
            u[3] += u[2];
            _mm256_storeu_pd(rv + k, _mm256_loadu_pd(u));
            carry = u[3];
          }
          for (int64_t k = n4; k < len; ++k){
            carry += nv[k] + rv[k];
            rv[k] = carry;
          }
        }
      }
    }
  }
}
