/* heat_3d v2: 4x4 (i,j) tiles streamed along k with a rolling 3-row register
 * window; AVX-512; OpenMP.
 *
 * The tile's halo rows (i0-1, i0+4, j0-1, j0+4) are loaded once per k-chunk
 * and shared by all 16 center rows of the tile, dropping DRAM traffic from
 * ~56 B/elem (1x1 rows) to ~32 B/elem. Edge strips (rows/cols not covered by
 * a full 4x4 tile) use the 1x1 row kernel.
 *
 * Per-element arithmetic order is identical to the numpy reference:
 *   ((alpha*((ip - 2*cc) + im) + alpha*((jp - 2*cc) + jm))
 *    + alpha*((kp - 2*cc) + km)) + cc
 * No a*b+c shapes -> no FMA contraction.
 */
#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>

static inline void row_kernel(const double *restrict rc, const double *restrict rim,
                              const double *restrict rip, const double *restrict rjm,
                              const double *restrict rjp, double *restrict ro,
                              int64_t L, int64_t k8, double alpha)
{
    const __m512d va = _mm512_set1_pd(alpha);
    const __m512d vt = _mm512_set1_pd(2.0);
    int64_t k = 0;
    for (; k < k8; k += 8) {
        __m512d v0 = _mm512_loadu_pd(rc + k);
        __m512d vm1 = _mm512_loadu_pd(rc + k - 1);
        __m512d vp1 = _mm512_loadu_pd(rc + k + 1);
        __m512d aip = _mm512_loadu_pd(rip + k);
        __m512d aim = _mm512_loadu_pd(rim + k);
        __m512d ajp = _mm512_loadu_pd(rjp + k);
        __m512d ajm = _mm512_loadu_pd(rjm + k);
        __m512d t1 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(aip, _mm512_mul_pd(vt, v0)), aim));
        __m512d t2 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(ajp, _mm512_mul_pd(vt, v0)), ajm));
        __m512d t3 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(vp1, _mm512_mul_pd(vt, v0)), vm1));
        _mm512_storeu_pd(ro + k, _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(t1, t2), t3), v0));
    }
    for (; k < L; ++k) {
        double cc = rc[k];
        double r1 = alpha * ((rip[k] - 2.0 * cc) + rim[k]);
        double r2 = alpha * ((rjp[k] - 2.0 * cc) + rjm[k]);
        double r3 = alpha * ((rc[k + 1] - 2.0 * cc) + rc[k - 1]);
        ro[k] = ((r1 + r2) + r3) + cc;
    }
}

/* one k-chunk for one output row; g0/g1/g2 hold the i-1/i/i+1 rows, each as 6
 * chunks of 8 doubles starting at column j0-1; jj indexes output column j0+jj */
static inline void out_row4(const __m512d *g0, const __m512d *g1, const __m512d *g2,
                            const double *restrict rc_row, double *restrict ro_row,
                            int64_t N, int64_t k, double alpha)
{
    const __m512d va = _mm512_set1_pd(alpha);
    const __m512d vt = _mm512_set1_pd(2.0);
    for (int jj = 0; jj < 4; ++jj) {
        const double *ccr = rc_row + (jj + 1) * N;
        __m512d v0 = g1[jj + 1];
        __m512d vm1 = _mm512_loadu_pd(ccr + k - 1);
        __m512d vp1 = _mm512_loadu_pd(ccr + k + 1);
        __m512d t1 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(g2[jj + 1], _mm512_mul_pd(vt, v0)),
                                                 g0[jj + 1]));
        __m512d t2 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(g1[jj + 2], _mm512_mul_pd(vt, v0)),
                                                 g1[jj]));
        __m512d t3 = _mm512_mul_pd(va,
                                   _mm512_add_pd(_mm512_sub_pd(vp1, _mm512_mul_pd(vt, v0)), vm1));
        _mm512_storeu_pd(ro_row + jj * N + k,
                         _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(t1, t2), t3), v0));
    }
}

/* 4x4 tile, interior i0..i0+3 / j0..j0+3 */
static inline void tile4(const double *restrict S, double *restrict D, int64_t N,
                         int64_t i0, int64_t j0, int64_t L, int64_t k8, double alpha)
{
    int64_t k = 0;
    for (; k < k8; k += 8) {
        const double *rws[6];
        for (int r = 0; r < 6; ++r)
            rws[r] = S + ((i0 - 1 + r) * N + (j0 - 1)) * N + 1;
        __m512d w[3][6];
        for (int r = 0; r < 3; ++r)
            for (int jj = 0; jj < 6; ++jj)
                w[r][jj] = _mm512_loadu_pd(rws[r] + jj * N + k);
        for (int ii = 0; ii < 4; ++ii) {
            out_row4(w[0], w[1], w[2], rws[ii + 1], D + ((i0 + ii) * N + j0) * N + 1, N, k, alpha);
            if (ii < 3) {
                for (int jj = 0; jj < 6; ++jj) {
                    w[0][jj] = w[1][jj];
                    w[1][jj] = w[2][jj];
                    w[2][jj] = _mm512_loadu_pd(rws[ii + 3] + jj * N + k);
                }
            }
        }
    }
    for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
            int64_t i = i0 + ii, j = j0 + jj;
            const double *restrict rc = S + (i * N + j) * N + 1;
            const double *restrict rim = S + ((i - 1) * N + j) * N + 1;
            const double *restrict rip = S + ((i + 1) * N + j) * N + 1;
            const double *restrict rjm = S + (i * N + j - 1) * N + 1;
            const double *restrict rjp = S + (i * N + j + 1) * N + 1;
            double *restrict ro = D + (i * N + j) * N + 1;
            for (int64_t kk = k8; kk < L; ++kk) {
                double cc = rc[kk];
                double r1 = alpha * ((rip[kk] - 2.0 * cc) + rim[kk]);
                double r2 = alpha * ((rjp[kk] - 2.0 * cc) + rjm[kk]);
                double r3 = alpha * ((rc[kk + 1] - 2.0 * cc) + rc[kk - 1]);
                ro[kk] = ((r1 + r2) + r3) + cc;
            }
        }
    }
}
#endif

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N,
                  int64_t TSTEPS, double alpha)
{
    if (N < 4 || TSTEPS <= 0)
        return;

    const int64_t L = N - 2;
    const int64_t k8 = (L >> 3) << 3;
    const int64_t nfull = L / 4;
    const int64_t iend_full = 1 + 4 * nfull;   /* first interior i NOT in tiles */
    const int64_t jend_full = 1 + 4 * nfull;   /* first interior j NOT in tiles */

#pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            /* sweep A -> B: 4x4 tiles */
#pragma omp for nowait schedule(static) collapse(2)
            for (int64_t a = 0; a < nfull; ++a) {
                for (int64_t b = 0; b < nfull; ++b) {
                    tile4(A, B, N, 1 + 4 * a, 1 + 4 * b, L, k8, alpha);
                }
            }
            /* j-strip: i in tiles, j beyond them */
#pragma omp for nowait schedule(static) collapse(2)
            for (int64_t i = 1; i < iend_full; ++i) {
                for (int64_t j = jend_full; j <= N - 2; ++j) {
                    int64_t ij = (i * N + j) * N;
                    row_kernel(A + ij + 1, A + ((i - 1) * N + j) * N + 1,
                               A + ((i + 1) * N + j) * N + 1, A + (i * N + j - 1) * N + 1,
                               A + (i * N + j + 1) * N + 1, B + ij + 1, L, k8, alpha);
                }
            }
            /* i-strip: i beyond tiles, all j */
#pragma omp for schedule(static) collapse(2)
            for (int64_t i = iend_full; i <= N - 2; ++i) {
                for (int64_t j = 1; j <= N - 2; ++j) {
                    int64_t ij = (i * N + j) * N;
                    row_kernel(A + ij + 1, A + ((i - 1) * N + j) * N + 1,
                               A + ((i + 1) * N + j) * N + 1, A + (i * N + j - 1) * N + 1,
                               A + (i * N + j + 1) * N + 1, B + ij + 1, L, k8, alpha);
                }
            }
            /* sweep B -> A: 4x4 tiles */
#pragma omp for nowait schedule(static) collapse(2)
            for (int64_t a = 0; a < nfull; ++a) {
                for (int64_t b = 0; b < nfull; ++b) {
                    tile4(B, A, N, 1 + 4 * a, 1 + 4 * b, L, k8, alpha);
                }
            }
#pragma omp for nowait schedule(static) collapse(2)
            for (int64_t i = 1; i < iend_full; ++i) {
                for (int64_t j = jend_full; j <= N - 2; ++j) {
                    int64_t ij = (i * N + j) * N;
                    row_kernel(B + ij + 1, B + ((i - 1) * N + j) * N + 1,
                               B + ((i + 1) * N + j) * N + 1, B + (i * N + j - 1) * N + 1,
                               B + (i * N + j + 1) * N + 1, A + ij + 1, L, k8, alpha);
                }
            }
#pragma omp for schedule(static) collapse(2)
            for (int64_t i = iend_full; i <= N - 2; ++i) {
                for (int64_t j = 1; j <= N - 2; ++j) {
                    int64_t ij = (i * N + j) * N;
                    row_kernel(B + ij + 1, B + ((i - 1) * N + j) * N + 1,
                               B + ((i + 1) * N + j) * N + 1, B + (i * N + j - 1) * N + 1,
                               B + (i * N + j + 1) * N + 1, A + ij + 1, L, k8, alpha);
                }
            }
        }
    }
}
