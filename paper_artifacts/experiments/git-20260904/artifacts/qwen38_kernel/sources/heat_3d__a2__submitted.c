/* Keep FP contraction OFF: the numpy oracle uses separate mul/add rounds, and
 * FMA contraction shifts results by ~1 ulp per element, which the time-stepped
 * stencil amplifies beyond tolerance. */
#pragma GCC optimize ("fp-contract=off")

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

/* 3D heat equation, 7-point stencil, interior update, two sweeps per step.
 * 2x2 blocking in (i, j) so neighboring rows are fetched once and reused from
 * L1/L3; k loop vectorized in aligned LANES-wide chunks. Exact FP op order of
 * the numpy reference:
 *   ((alpha*((iP-2c)+iM)) + (alpha*((jP-2c)+jM))) + (alpha*((kP-2c)+kM)) + c
 */
static inline double stl(double c, double iP, double iM, double jP, double jM, double kP, double kM, double alpha) {
    double t0 = alpha * ((iP - 2.0 * c) + iM);
    double t1 = alpha * ((jP - 2.0 * c) + jM);
    double t2 = alpha * ((kP - 2.0 * c) + kM);
    return ((t0 + t1) + t2) + c;
}

#if defined(__AVX512F__)
#define LANES 8
static __inline__ void rowv(const double *restrict c, const double *restrict iP, const double *restrict iM,
                            const double *restrict jP, const double *restrict jM,
                            double *restrict dst, double alpha, int64_t n) {
    __m512d va = _mm512_set1_pd(alpha);
    __m512d v2 = _mm512_set1_pd(2.0);
    int64_t k = 0;
    for (; k + LANES <= n; k += LANES) {
        __m512d vc  = _mm512_loadu_pd(c + k);
        __m512d two = _mm512_mul_pd(v2, vc);
        __m512d t0 = _mm512_mul_pd(va, _mm512_add_pd(_mm512_sub_pd(_mm512_loadu_pd(iP + k), two), _mm512_loadu_pd(iM + k)));
        __m512d t1 = _mm512_mul_pd(va, _mm512_add_pd(_mm512_sub_pd(_mm512_loadu_pd(jP + k), two), _mm512_loadu_pd(jM + k)));
        __m512d t2 = _mm512_mul_pd(va, _mm512_add_pd(_mm512_sub_pd(_mm512_loadu_pd(c + k + 1), two), _mm512_loadu_pd(c + k - 1)));
        _mm512_storeu_pd(dst + k, _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(t0, t1), t2), vc));
    }
    for (; k < n; k++)
        dst[k] = stl(c[k], iP[k], iM[k], jP[k], jM[k], c[k + 1], c[k - 1], alpha);
}
#elif defined(__AVX2__)
#define LANES 4
static __inline__ void rowv(const double *restrict c, const double *restrict iP, const double *restrict iM,
                            const double *restrict jP, const double *restrict jM,
                            double *restrict dst, double alpha, int64_t n) {
    __m256d va = _mm256_set1_pd(alpha);
    __m256d v2 = _mm256_set1_pd(2.0);
    int64_t k = 0;
    for (; k + LANES <= n; k += LANES) {
        __m256d vc  = _mm256_loadu_pd(c + k);
        __m256d two = _mm256_mul_pd(v2, vc);
        __m256d t0 = _mm256_mul_pd(va, _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(iP + k), two), _mm256_loadu_pd(iM + k)));
        __m256d t1 = _mm256_mul_pd(va, _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(jP + k), two), _mm256_loadu_pd(jM + k)));
        __m256d t2 = _mm256_mul_pd(va, _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(c + k + 1), two), _mm256_loadu_pd(c + k - 1)));
        _mm256_storeu_pd(dst + k, _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(t0, t1), t2), vc));
    }
    for (; k < n; k++)
        dst[k] = stl(c[k], iP[k], iM[k], jP[k], jM[k], c[k + 1], c[k - 1], alpha);
}
#else
#define LANES 1
static __inline__ void rowv(const double *restrict c, const double *restrict iP, const double *restrict iM,
                            const double *restrict jP, const double *restrict jM,
                            double *restrict dst, double alpha, int64_t n) {
    for (int64_t k = 0; k < n; k++)
        dst[k] = stl(c[k], iP[k], iM[k], jP[k], jM[k], c[k + 1], c[k - 1], alpha);
}
#endif

static __inline__ void onerow(const double *restrict S, double *restrict D, int64_t N, double alpha,
                              int64_t I, int64_t J, int64_t N2) {
    const double *c  = S + (I * N + J) * N;
    const double *iP = S + ((I + 1) * N + J) * N;
    const double *iM = S + ((I - 1) * N + J) * N;
    const double *jP = S + (I * N + (J + 1)) * N;
    const double *jM = S + (I * N + (J - 1)) * N;
    double *d = D + (I * N + J) * N;
    for (int64_t k = 1; k < LANES && k <= N2; k++)
        d[k] = stl(c[k], iP[k], iM[k], jP[k], jM[k], c[k + 1], c[k - 1], alpha);
    if (N2 >= LANES)
        rowv(c + LANES, iP + LANES, iM + LANES, jP + LANES, jM + LANES, d + LANES, alpha, N2 - LANES + 1);
}

static void sweep(const double *restrict S, double *restrict D, int64_t N, double alpha) {
    int64_t N2 = N - 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i0 = 1; i0 <= N2; i0 += 2) {
        for (int64_t j0 = 1; j0 <= N2; j0 += 2) {
            onerow(S, D, N, alpha, i0, j0, N2);
            if (j0 + 1 <= N2) onerow(S, D, N, alpha, i0, j0 + 1, N2);
            if (i0 + 1 <= N2) {
                onerow(S, D, N, alpha, i0 + 1, j0, N2);
                if (j0 + 1 <= N2) onerow(S, D, N, alpha, i0 + 1, j0 + 1, N2);
            }
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    for (int64_t t = 0; t < TSTEPS; t++) {
        sweep(A, B, N, alpha);
        sweep(B, A, N, alpha);
    }
}
