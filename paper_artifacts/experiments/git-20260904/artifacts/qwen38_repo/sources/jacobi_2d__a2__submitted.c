/* 2-D Jacobi 5-point stencil, ping-pong A/B, TSTEPS sweeps.
 *
 * Optimized:
 *  - OpenMP: one persistent team for the whole kernel; row-parallel
 *    work-share on each half-step (rows are uniform work => static).
 *  - AVX-512 (when compiled with -march=native on an AVX512F CPU):
 *    8 doubles/iter, exact same per-element op order as the reference
 *    ((c+l)+r+b+t)*0.2 so results are bit-identical to scalar code.
 *  - Scalar tail for odd widths (e.g. N-2 not a multiple of 8).
 *
 * C-ABI symbol and signature unchanged: jacobi_2d_fp64(A, B, N, TSTEPS).
 */
#include <stdint.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* One half-step over rows [0, N): compute dst interior from src.
 * Rows are independent; the caller parallelizes the row range. */
static void stencil_rows(const double *restrict src, double *restrict dst,
                         int64_t N, int64_t i_begin, int64_t i_end)
{
    const int64_t n8_end = N - 9; /* vector while j + 8 <= N - 1 */
    for (int64_t i = i_begin; i <= i_end; ++i) {
        const double *restrict rm = src + (i - 1) * N;
        const double *restrict rc = src + i * N;
        const double *restrict rp = src + (i + 1) * N;
        double *restrict dr = dst + i * N;
        int64_t j = 1;
#if defined(__AVX512F__)
        for (; j <= n8_end; j += 8) {
            __m512d c = _mm512_loadu_pd(rc + j);
            __m512d s = _mm512_add_pd(c, _mm512_loadu_pd(rc + j - 1));
            s = _mm512_add_pd(s, _mm512_loadu_pd(rc + j + 1));
            s = _mm512_add_pd(s, _mm512_loadu_pd(rp + j));
            s = _mm512_add_pd(s, _mm512_loadu_pd(rm + j));
            _mm512_storeu_pd(dr + j, _mm512_mul_pd(s, _mm512_set1_pd(0.2)));
        }
#endif
        for (; j < N - 1; ++j)
            dr[j] = 0.2 * ((((rc[j] + rc[j - 1]) + rc[j + 1]) + rp[j]) + rm[j]);
    }
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N,
                    int64_t TSTEPS)
{
    if (N < 3 || TSTEPS < 1)
        return;

    const int64_t i_end = N - 2;

#pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
#pragma omp for schedule(static) nowait
            for (int64_t i = 1; i <= i_end; ++i)
                stencil_rows(B, A, N, i, i);
#pragma omp for schedule(static)
            for (int64_t i = 1; i <= i_end; ++i)
                stencil_rows(A, B, N, i, i);
        }
    }
}
