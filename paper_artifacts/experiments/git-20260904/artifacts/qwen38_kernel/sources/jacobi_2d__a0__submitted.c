#include <stdint.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

#if HAVE_AVX512
#define C02 _mm512_set1_pd(0.2)
/* Exact reference association: 0.2 * (((((c + l) + r) + d) + u)) */
#define ST4(C, L, R, D, U)                                             \
    _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(                         \
        _mm512_add_pd(_mm512_add_pd((C), (L)), (R)), (D)), (U)), C02)
#endif

/* Scalar single-row stencil (exact reference order). */
static inline void row_scalar(const double *restrict up, const double *restrict mid,
                              const double *restrict dn, double *restrict out,
                              int64_t N)
{
    for (int64_t j = 1; j < N - 1; ++j)
        out[j] = 0.2 * (mid[j] + mid[j - 1] + mid[j + 1] + dn[j] + up[j]);
}

#if HAVE_AVX512
/* SIMD single-row stencil: 8 doubles per iteration. */
static inline void row_simd(const double *restrict up, const double *restrict mid,
                            const double *restrict dn, double *restrict out,
                            int64_t N)
{
    int64_t j = 1;
    for (; j + 8 <= N - 1; j += 8) {
        __m512d c = _mm512_loadu_pd(mid + j);
        __m512d l = _mm512_loadu_pd(mid + j - 1);
        __m512d r = _mm512_loadu_pd(mid + j + 1);
        __m512d d = _mm512_loadu_pd(dn + j);
        __m512d u = _mm512_loadu_pd(up + j);
        _mm512_storeu_pd(out + j, ST4(c, l, r, d, u));
    }
    for (; j < N - 1; ++j)
        out[j] = 0.2 * (mid[j] + mid[j - 1] + mid[j + 1] + dn[j] + up[j]);
}

/* Two consecutive output rows in one j-loop: the shared middle row is
 * loaded once per column block (5 row streams, fits the L1 set bank where
 * 4KB-spaced rows collide; 4-row groups conflict and thrash). */
static inline void rows2_simd(const double *restrict IN, double *restrict OUT,
                              int64_t N, int64_t i)
{
    const double *r0 = IN + (i - 1) * N;   /* above row i */
    const double *m0 = r0 + N;             /* rows i, i+1 */
    const double *m1 = m0 + N;
    const double *m2 = m1 + N;             /* below row i+1 */
    double *o0 = OUT + i * N;
    double *o1 = o0 + N;
    int64_t j = 1;
    for (; j + 8 <= N - 1; j += 8) {
        __m512d u0  = _mm512_loadu_pd(r0 + j);
        __m512d m0l = _mm512_loadu_pd(m0 + j - 1);
        __m512d m0c = _mm512_loadu_pd(m0 + j);
        __m512d m0r = _mm512_loadu_pd(m0 + j + 1);
        __m512d m1c = _mm512_loadu_pd(m1 + j);
        _mm512_storeu_pd(o0 + j, ST4(m0c, m0l, m0r, m1c, u0));
        __m512d m1l = _mm512_loadu_pd(m1 + j - 1);
        __m512d m1r = _mm512_loadu_pd(m1 + j + 1);
        __m512d m2c = _mm512_loadu_pd(m2 + j);
        _mm512_storeu_pd(o1 + j, ST4(m1c, m1l, m1r, m2c, m0c));
    }
    for (; j < N - 1; ++j)
        for (int64_t k = 0; k < 2; ++k)
            OUT[(i + k) * N + j] =
                0.2 * (IN[(i + k) * N + j] + IN[(i + k) * N + j - 1] +
                       IN[(i + k) * N + j + 1] + IN[(i + k + 1) * N + j] +
                       IN[(i + k - 1) * N + j]);
}
#endif

/* Rows [r0, r1) of OUT from IN: groups of 2 rows SIMD, scalar fallback. */
static inline void pass_rows(const double *restrict IN, double *restrict OUT,
                             int64_t N, int64_t r0, int64_t r1)
{
#if HAVE_AVX512
    int64_t i = r0;
    for (; i + 2 <= r1; i += 2)
        rows2_simd(IN, OUT, N, i);
    for (; i < r1; ++i)
        row_simd(IN + (i - 1) * N, IN + i * N, IN + (i + 1) * N, OUT + i * N, N);
#else
    for (int64_t i = r0; i < r1; ++i)
        row_scalar(IN + (i - 1) * N, IN + i * N, IN + (i + 1) * N, OUT + i * N, N);
#endif
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS)
{
    if (N < 4 || TSTEPS <= 0) return;
    const int64_t ngr = (N - 2 + 1) / 2;   /* number of 2-row groups */
    #pragma omp parallel
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp for schedule(static)
        for (int64_t g = 0; g < ngr; ++g) {
            int64_t i0 = 1 + 2 * g;
            int64_t i1 = i0 + 2;
            if (i1 > N - 1) i1 = N - 1;
            pass_rows(A, B, N, i0, i1);
        }
        #pragma omp for schedule(static)
        for (int64_t g = 0; g < ngr; ++g) {
            int64_t i0 = 1 + 2 * g;
            int64_t i1 = i0 + 2;
            if (i1 > N - 1) i1 = N - 1;
            pass_rows(B, A, N, i0, i1);
        }
    }
}
