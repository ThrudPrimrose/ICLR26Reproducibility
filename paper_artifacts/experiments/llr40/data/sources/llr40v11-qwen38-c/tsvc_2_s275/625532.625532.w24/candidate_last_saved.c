/* tsvc_2_s275: per-column prefix scan, vectorized across columns (AVX2).
 * aa[j*N+i] = aa[(j-1)*N+i] + bb[j*N+i]*cc[j*N+i]   (only if aa[0*N+i] > 0)
 * Columns are independent -> OpenMP over column blocks; the serial carry is
 * kept in one register per 256-bit lane (lane = column). Exact mul-then-add
 * (no FMA contraction) to bit-match the NumPy reference rounding.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#pragma GCC optimize("fp-contract=off")
void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return;

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i0 = 0; i0 < N; i0 += 16) {
        const int ncol = (N - i0 < 16) ? (int)(N - i0) : 16;
        const double *const a0 = aa + i0;   /* row 0, this block's columns */
        int act[16];
        int nact = 0;
        for (int k = 0; k < ncol; k++) { act[k] = (a0[k] > 0.0) ? 1 : 0; nact += act[k]; }
        if (nact == 0) continue;

        const int nvec = ncol >> 2;

        __m256d carry[4], maskv[4];
        double scarry[4];
        for (int g = 0; g < nvec; g++) {
            carry[g] = _mm256_loadu_pd(a0 + (g << 2));
            maskv[g] = _mm256_cmp_pd(carry[g], _mm256_setzero_pd(), _CMP_GT_OQ);
        }
        for (int k = nvec << 2; k < ncol; k++) scarry[k] = a0[k];

        if (nact == ncol) {
            /* all columns active: pure mul+add chain */
            for (int64_t j = 1; j < N; j++) {
                const double *const brow = bb + j * N + i0;
                const double *const crow = cc + j * N + i0;
                double *const arow = aa + j * N + i0;
                #pragma GCC unroll 4
                for (int g = 0; g < nvec; g++) {
                    const __m256d b = _mm256_loadu_pd(brow + (g << 2));
                    const __m256d c = _mm256_loadu_pd(crow + (g << 2));
                    const __m256d t = _mm256_mul_pd(b, c);
                    __m256d v = _mm256_add_pd(carry[g], t);
                    carry[g] = v;
                    _mm256_storeu_pd(arow + (g << 2), v);
                }
                for (int k = nvec << 2; k < ncol; k++) {
                    const double t = brow[k] * crow[k];
                    scarry[k] += t;
                    arow[k] = scarry[k];
                }
            }
        } else {
            /* mixed: inactive columns keep their original values (blend) */
            for (int64_t j = 1; j < N; j++) {
                const double *const brow = bb + j * N + i0;
                const double *const crow = cc + j * N + i0;
                double *const arow = aa + j * N + i0;
                #pragma GCC unroll 4
                for (int g = 0; g < nvec; g++) {
                    const __m256d b = _mm256_loadu_pd(brow + (g << 2));
                    const __m256d c = _mm256_loadu_pd(crow + (g << 2));
                    const __m256d orig = _mm256_loadu_pd(arow + (g << 2));
                    const __m256d t = _mm256_mul_pd(b, c);
                    const __m256d u = _mm256_add_pd(carry[g], t);
                    const __m256d v = _mm256_blendv_pd(orig, u, maskv[g]);
                    carry[g] = v;
                    _mm256_storeu_pd(arow + (g << 2), v);
                }
                for (int k = nvec << 2; k < ncol; k++) {
                    if (act[k]) {
                        const double t = brow[k] * crow[k];
                        scarry[k] += t;
                        arow[k] = scarry[k];
                    }
                }
            }
        }
    }
}
