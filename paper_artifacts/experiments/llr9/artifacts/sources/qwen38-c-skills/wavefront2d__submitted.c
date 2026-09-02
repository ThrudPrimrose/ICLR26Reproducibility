/* hpcagent_bench stub headers -- keep as given */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#include <immintrin.h>

/*
 * In-place 2D relaxation (Gauss-Seidel sweep in the reference):
 *   for i in 1..n-1: for j in 1..n-1:
 *       a[i][j] = 0.25*(a[i][j] + a[i-1][j] + a[i][j-1] + a[i-1][j-1])
 *
 * (i,j) reads (i-1,j) and (i,j-1) -- BOTH axes carry a dependence, so the
 * only parallel levels are the anti-diagonals k = i+j (wavefront).  Both the
 * row-major sweep and any diagonal-ordered execution are linear extensions of
 * the same partial order, so the result is bit-identical either way.
 *
 * Tiled wavefront: w x w blocks; block anti-diagonals (bi+bj = R) are
 * independent of each other, so one round = one block anti-diagonal with a
 * single barrier.  Inside a block a mini-wavefront over its 2w-2 diagonals;
 * cross-block reads (north/west/NW rows) are already final from rounds R-1/R-2.
 *
 * Inside a (block) diagonal the memory stride is (n-1) -> strided gather /
 * scalar tail, 8-wide with AVX-512.
 */

static inline void seg_update(double *restrict a, const int64_t n, const int64_t k,
                              int64_t i_lo, int64_t i_hi)
{
    const int64_t w = n - 1; /* a[i][k-i] = a[i*w + k] */
#if defined(__AVX512F__)
    if (i_hi - i_lo >= 7) {
        const int64_t iv = i_hi - ((i_hi - i_lo) & 7);
        const __m512i voff =
            _mm512_setr_epi64(0, w, 2 * w, 3 * w, 4 * w, 5 * w, 6 * w, 7 * w);
        const __m512i vn   = _mm512_sub_epi64(voff, _mm512_set1_epi64(n));
        const __m512i v1   = _mm512_sub_epi64(voff, _mm512_set1_epi64(1));
        const __m512i vn1  = _mm512_sub_epi64(vn, _mm512_set1_epi64(1));
        const double *b = a + i_lo * w + k;
        for (int64_t i = i_lo; i < iv; i += 8, b += 8 * w) {
            __m512d s = _mm512_i64gather_pd(voff, b, 8);
            s = _mm512_add_pd(s, _mm512_i64gather_pd(vn, b, 8));
            s = _mm512_add_pd(s, _mm512_i64gather_pd(v1, b, 8));
            s = _mm512_add_pd(s, _mm512_add_pd(_mm512_i64gather_pd(vn1, b, 8),
                                               _mm512_set1_pd(0.0)));
            _mm512_i64scatter_pd(b, voff, _mm512_mul_pd(s, _mm512_set1_pd(0.25)), 8);
        }
        i_lo = iv;
    }
#endif
    for (int64_t i = i_lo; i <= i_hi; i++) {
        const int64_t b = i * w + k;
        a[b] = 0.25 * (a[b] + a[b - n] + a[b - 1] + a[b - n - 1]);
    }
}

/* wavefront over rectangle rows [r0,r1), cols [c0,c1); cells with r>=1, c>=1 */
static inline void block_update(double *restrict a, const int64_t n,
                                int64_t r0, int64_t r1, int64_t c0, int64_t c1)
{
    const int64_t kmax = (r1 - 1) + (c1 - 1);
    for (int64_t k = r0 + c0; k <= kmax; k++) {
        int64_t i_lo = k - (c1 - 1);
        if (i_lo < r0) i_lo = r0;
        if (i_lo < 1) i_lo = 1;
        int64_t i_hi = k - c0;
        if (i_hi > (int64_t)(r1 - 1)) i_hi = r1 - 1;
        if (i_hi > n - 1) i_hi = n - 1;
        if (i_hi > k - 1) i_hi = k - 1;
        if (i_lo <= i_hi) seg_update(a, n, k, i_lo, i_hi);
    }
}

void wavefront2d_fp64(double *restrict a, const int64_t LEN_2D,
                      uint8_t *restrict workspace, const int64_t workspace_size)
{
    const int64_t n = LEN_2D;
    (void)workspace;
    (void)workspace_size;
    if (n < 2) return;

    const int64_t nt = omp_get_max_threads() > 0 ? omp_get_max_threads() : 1;

    printf("wavefront2d_probe n=%lld nt=%lld\n", (long long)n, (long long)nt);
    fflush(stdout);

    if (nt < 2 || n < 512) {
        block_update(a, n, 0, n, 0, n);
        return;
    }

    const int64_t w = (n >= 2048) ? 64 : 32;
    const int64_t nb = (n + w - 1) / w; /* blocks per side */

    #pragma omp parallel
    {
        for (int64_t R = 0; R <= 2 * nb - 2; R++) {
            int64_t bi_lo = R < nb ? 0 : R - (nb - 1);
            int64_t bi_hi = R < nb - 1 ? R : nb - 1;
            int64_t cnt = bi_hi - bi_lo + 1;
            #pragma omp for schedule(static)
            for (int64_t t = 0; t < cnt; t++) {
                int64_t bi = bi_lo + t;
                int64_t bj = R - bi;
                int64_t r0 = bi * w;
                int64_t r1 = r0 + w; if (r1 > n) r1 = n;
                int64_t c0 = bj * w;
                int64_t c1 = c0 + w; if (c1 > n) c1 = n;
                block_update(a, n, r0, r1, c0, c1);
            }
        }
    }
}
