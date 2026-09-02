#include <stdint.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

/* LU without pivoting (right-looking) + triangular solves.
 * Per-k: scale multipliers below the diagonal and apply the rank-1 row
 * update; rows are distributed statically across a persistent team.
 * The inner update uses a hand-unrolled 4x512-bit FMA block per 32
 * doubles (2 rows interleaved) to hide load/store latency; per-element
 * arithmetic (division + fused multiply-subtract) matches the reference. */

#if HAS_AVX512
static inline void row_update32(double *restrict Ai, const double *restrict Ak,
                                double l, int64_t j, int64_t N)
{
    __m512d vl = _mm512_set1_pd(l);
    const int64_t jlim = N - 32;
    for (; j <= jlim; j += 32) {
        __m512d v0 = _mm512_loadu_pd(Ai + j);
        __m512d v1 = _mm512_loadu_pd(Ai + j + 8);
        __m512d v2 = _mm512_loadu_pd(Ai + j + 16);
        __m512d v3 = _mm512_loadu_pd(Ai + j + 24);
        __m512d u0 = _mm512_loadu_pd(Ak + j);
        __m512d u1 = _mm512_loadu_pd(Ak + j + 8);
        __m512d u2 = _mm512_loadu_pd(Ak + j + 16);
        __m512d u3 = _mm512_loadu_pd(Ak + j + 24);
        _mm512_storeu_pd(Ai + j,      _mm512_fnmadd_pd(u0, vl, v0));
        _mm512_storeu_pd(Ai + j + 8,  _mm512_fnmadd_pd(u1, vl, v1));
        _mm512_storeu_pd(Ai + j + 16, _mm512_fnmadd_pd(u2, vl, v2));
        _mm512_storeu_pd(Ai + j + 24, _mm512_fnmadd_pd(u3, vl, v3));
    }
    for (; j < N; ++j) Ai[j] = Ai[j] - l * Ak[j];
}
#endif

static void factor_serial(double *restrict A, int64_t N)
{
    for (int64_t k = 0; k < N; ++k) {
        const double *restrict Ak = A + k * N;
        const double diag = Ak[k];
        for (int64_t i = k + 1; i < N; ++i) {
            double *restrict Ai = A + i * N;
            const double l = Ai[k] / diag;
            Ai[k] = l;
            int64_t j = k + 1;
#if HAS_AVX512
            row_update32(Ai, Ak, l, j, N);
#else
            for (; j < N; ++j) Ai[j] = Ai[j] - l * Ak[j];
#endif
        }
    }
}

static void factor_parallel(double *restrict A, int64_t N)
{
    int64_t t = (int64_t)omp_get_max_threads();
    int64_t cap = N / 16;
    if (cap < 4) cap = 4;
    if (t > cap) t = cap;
    #pragma omp parallel num_threads((int)t)
    for (int64_t k = 0; k < N; ++k) {
        const double diag = A[k * N + k];
        #pragma omp for schedule(static)
        for (int64_t i = k + 1; i < N; ++i) {
            double *restrict Ai = A + i * N;
            const double *restrict Ak = A + k * N;
            const double l = Ai[k] / diag;
            Ai[k] = l;
#if HAS_AVX512
            row_update32(Ai, Ak, l, k + 1, N);
#else
            for (int64_t j = k + 1; j < N; ++j) {
                Ai[j] = Ai[j] - l * Ak[j];
            }
#endif
        }
    }
}

void ludcmp_fp64(double *restrict A, const double *restrict b,
                 double *restrict x, double *restrict y, int64_t N)
{
    if (N < 96)
        factor_serial(A, N);
    else
        factor_parallel(A, N);

    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const double *restrict Ai = A + i * N;
        for (int64_t j = 0; j < i; ++j) s += Ai[j] * y[j];
        y[i] = b[i] - s;
    }
    for (int64_t i = N - 1; i >= 0; --i) {
        double s = 0.0;
        const double *restrict Ai = A + i * N;
        for (int64_t j = i + 1; j < N; ++j) s += Ai[j] * x[j];
        x[i] = (y[i] - s) / Ai[i];
    }
}
