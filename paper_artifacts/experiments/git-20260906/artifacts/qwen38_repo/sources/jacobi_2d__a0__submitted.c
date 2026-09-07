/* Optimized 2-D Jacobi stencil (5-point, double-buffered).
 *
 * The timestep loop is a recurrence, but inside one half-step every interior
 * row is independent (reads only src, writes only dst), so rows are farmed
 * to one persistent OpenMP team; static scheduling keeps each thread's row
 * block cache-hot across timesteps.  The column loop is vectorized
 * (AVX-512 where available, else AVX2, else scalar): three overlapping loads
 * of the center row supply left/center/right, plus one load each for the top
 * and bottom rows -> 5 FMA-pipe ops per 8 elements, with the reference
 * rounding order 0.2 * ((((c+l)+r)+d)+u) preserved exactly.
 *
 * When the working set exceeds the LLC, outputs are written with 16B
 * non-temporal stores (vmovnt) so the store stream completes whole 64B lines
 * in L2 and is written back to DRAM without a read-for-ownership.  The vector
 * grid is shifted by one column on rows where that makes (row+j) even so the
 * 16B stores stay 16-byte aligned.
 */
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#define JAC_CONST 0.2

#if defined(__AVX512F__)
static inline void row_simd(const double *restrict src, double *restrict dst,
                            int64_t i, int64_t N, int use_nt) {
    const double *restrict t = src + (i - 1) * N;
    const double *restrict c = src + i * N;
    const double *restrict b = src + (i + 1) * N;
    double *restrict d = dst + i * N;
    const __m512d k = _mm512_set1_pd(JAC_CONST);
    int64_t j = 1;
    if (use_nt) {
        /* start the 8-lane grid where (i*N + j) % 4 == 0 so the two 32B
         * non-temporal stores stay 32-byte aligned */
        int64_t j0 = ((4 - (i * N) % 4) % 4);
        if (j0 < 1) j0 += 4;
        while (j < j0) {
            d[j] = JAC_CONST * ((((c[j] + c[j - 1]) + c[j + 1]) + b[j]) + t[j]);
            ++j;
        }
    }
    for (; j + 8 <= N - 1; j += 8) {
        __m512d vc = _mm512_loadu_pd(c + j);
        __m512d vl = _mm512_loadu_pd(c + j - 1);
        __m512d vr = _mm512_loadu_pd(c + j + 1);
        __m512d vt = _mm512_loadu_pd(t + j);
        __m512d vb = _mm512_loadu_pd(b + j);
        __m512d s = _mm512_add_pd(vc, vl);
        s = _mm512_add_pd(s, vr);
        s = _mm512_add_pd(s, vb);
        s = _mm512_add_pd(s, vt);
        __m512d v = _mm512_mul_pd(s, k);
        if (use_nt) {
            _mm256_stream_pd(d + j, _mm512_castpd512_pd256(v));
            _mm256_stream_pd(d + j + 4, _mm512_extractf64x4_pd(v, 1));
        } else {
            _mm512_storeu_pd(d + j, v);
        }
    }
    for (; j < N - 1; ++j)
        d[j] = JAC_CONST * ((((c[j] + c[j - 1]) + c[j + 1]) + b[j]) + t[j]);
}
#elif defined(__AVX2__)
static inline void row_simd(const double *restrict src, double *restrict dst,
                            int64_t i, int64_t N, int use_nt) {
    (void)use_nt;
    const double *restrict t = src + (i - 1) * N;
    const double *restrict c = src + i * N;
    const double *restrict b = src + (i + 1) * N;
    double *restrict d = dst + i * N;
    const __m256d k = _mm256_set1_pd(JAC_CONST);
    int64_t j = 1;
    for (; j + 4 <= N - 1; j += 4) {
        __m256d vc = _mm256_loadu_pd(c + j);
        __m256d vl = _mm256_loadu_pd(c + j - 1);
        __m256d vr = _mm256_loadu_pd(c + j + 1);
        __m256d vt = _mm256_loadu_pd(t + j);
        __m256d vb = _mm256_loadu_pd(b + j);
        __m256d s = _mm256_add_pd(vc, vl);
        s = _mm256_add_pd(s, vr);
        s = _mm256_add_pd(s, vb);
        s = _mm256_add_pd(s, vt);
        _mm256_storeu_pd(d + j, _mm256_mul_pd(s, k));
    }
    for (; j < N - 1; ++j)
        d[j] = JAC_CONST * ((((c[j] + c[j - 1]) + c[j + 1]) + b[j]) + t[j]);
}
#else
static inline void row_simd(const double *restrict src, double *restrict dst,
                            int64_t i, int64_t N, int use_nt) {
    (void)use_nt;
    const double *restrict t = src + (i - 1) * N;
    const double *restrict c = src + i * N;
    const double *restrict b = src + (i + 1) * N;
    double *restrict d = dst + i * N;
    for (int64_t j = 1; j < N - 1; ++j)
        d[j] = JAC_CONST * ((((c[j] + c[j - 1]) + c[j + 1]) + b[j]) + t[j]);
}
#endif

/* Enough rows per thread to outweigh the sync, no more threads than cores. */
static int choose_threads(int64_t rows) {
    int maxt = omp_get_max_threads();
    int want = (int)(rows / 8);
    if (want < 1) want = 1;
    if (want > maxt) want = maxt;
    return want;
}

void jacobi_2d_fp64(double *restrict A, double *restrict B,
                    const int64_t N, const int64_t TSTEPS) {
    if (N <= 2 || TSTEPS <= 0) return;
    const int64_t rows = N - 2;
    const int P = choose_threads(rows);
    /* Non-temporal stores when the working set streams from DRAM. */
    const int use_nt = (N * N * 16) > (384LL << 20);

#pragma omp parallel num_threads(P)
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
#pragma omp for schedule(static, 1) nowait
            for (int64_t i = 1; i < N - 1; ++i) row_simd(A, B, i, N, use_nt);
#pragma omp barrier
#pragma omp for schedule(static, 1)
            for (int64_t i = 1; i < N - 1; ++i) row_simd(B, A, i, N, use_nt);
        }
    }
}
