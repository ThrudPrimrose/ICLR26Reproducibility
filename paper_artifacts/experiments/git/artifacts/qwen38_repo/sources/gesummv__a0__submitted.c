/* hpcagent_bench-autogen -- generated from gesummv_numpy.py; edited as a hand override.
 *
 * out[i] = alpha * (A[i,:] . x) + beta * (B[i,:] . x)
 *
 * Optimizations vs the naive baseline:
 *  - no temporary arrays (the baseline materialized alpha*A and beta*B),
 *  - the two mat-vecs are fused into one pass over the same cache lines,
 *  - explicit AVX-512 FMA dot products (4x unrolled, 64 doubles per step,
 *    x loaded once per 16 elements and reused for both A and B),
 *  - software prefetch of A/B rows ahead of the FMA pipeline,
 *  - rows distributed across OpenMP threads with static scheduling.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

static inline double reduce512(__m512d v) {
    __m256d lo = _mm512_castpd512_pd256(v);        /* lanes 0-3 */
    __m256d hi = _mm512_extractf64x4_pd(v, 1);     /* lanes 4-7 */
    __m256d s = _mm256_add_pd(lo, hi);             /* 2 sums of 4 lanes */
    __m128d r = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
    return _mm_cvtsd_f64(r);
}

void gesummv_fp64(const double *restrict A, const double *restrict B, double *restrict out,
                  const double *restrict x, const int64_t N, const double alpha, const double beta) {
    const int64_t nvec = N - (N & 63); /* longest 4*16-wide prefix of each row */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) {
        const double *restrict ar = A + (size_t)i * (size_t)N;
        const double *restrict br = B + (size_t)i * (size_t)N;

        __m512d va0 = _mm512_setzero_pd(), va1 = _mm512_setzero_pd(),
               va2 = _mm512_setzero_pd(), va3 = _mm512_setzero_pd();
        __m512d vb0 = _mm512_setzero_pd(), vb1 = _mm512_setzero_pd(),
               vb2 = _mm512_setzero_pd(), vb3 = _mm512_setzero_pd();

        int64_t j = 0;
        for (; j < nvec; j += 64) {
            const double *ap = ar + j, *bp = br + j, *xp = x + j;
            __builtin_prefetch(ap + 64, 0, 3);
            __builtin_prefetch(bp + 64, 0, 3);
            __m512d x0 = _mm512_loadu_pd(xp);
            __m512d x1 = _mm512_loadu_pd(xp + 16);
            __m512d x2 = _mm512_loadu_pd(xp + 32);
            __m512d x3 = _mm512_loadu_pd(xp + 48);
            va0 = _mm512_fmadd_pd(_mm512_loadu_pd(ap),     x0, va0);
            vb0 = _mm512_fmadd_pd(_mm512_loadu_pd(bp),     x0, vb0);
            va1 = _mm512_fmadd_pd(_mm512_loadu_pd(ap + 16), x1, va1);
            vb1 = _mm512_fmadd_pd(_mm512_loadu_pd(bp + 16), x1, vb1);
            va2 = _mm512_fmadd_pd(_mm512_loadu_pd(ap + 32), x2, va2);
            vb2 = _mm512_fmadd_pd(_mm512_loadu_pd(bp + 32), x2, vb2);
            va3 = _mm512_fmadd_pd(_mm512_loadu_pd(ap + 48), x3, va3);
            vb3 = _mm512_fmadd_pd(_mm512_loadu_pd(bp + 48), x3, vb3);
        }
        double da = reduce512(_mm512_add_pd(_mm512_add_pd(va0, va1), _mm512_add_pd(va2, va3)));
        double db = reduce512(_mm512_add_pd(_mm512_add_pd(vb0, vb1), _mm512_add_pd(vb2, vb3)));
        for (; j < N; j++) {
            da += ar[j] * x[j];
            db += br[j] * x[j];
        }
        out[i] = alpha * da + beta * db;
    }
}
