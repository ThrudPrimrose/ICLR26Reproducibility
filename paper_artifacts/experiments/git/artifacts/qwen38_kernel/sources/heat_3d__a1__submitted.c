#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* 3D heat equation, 7-point stencil, ping-pong A<->B, TSTEPS steps (2 sweeps each).
 * out[i,j,k] = c1*in[i,j,k] + c2*(in[i+1,j,k]+in[i-1,j,k]+in[i,j+1,k]+in[i,j-1,k]+in[i,j,k+1]+in[i,j,k-1])
 *
 * AVX-512 over k. Work is divided into groups of 4 adjacent i-slabs per thread:
 * the slab-neighbor rows of the middle slabs are each other's center rows (a/b
 * = e of neighbor, L1-resident), so each input row is fetched once per group
 * instead of once per slab -- roughly halves the L1-miss traffic vs a per-slab
 * sweep. The e row of each slab is 64B-aligned (per-row k0); a/b/c/d are
 * unaligned loads; k-neighbors via lane shifts. No masked loads anywhere
 * (Zen4 spurious-#GP erratum for partial-mask 512b loads across 64B lines).
 */

static inline double dot4(const double *a, const double *b, const double *c,
                          const double *d, const double *e, int64_t k,
                          double c1, double c2)
{
    return c1 * e[k] + c2 * (a[k] + b[k] + c[k] + d[k] + e[k - 1] + e[k + 1]);
}

/* 8-point window on e at k; e+k is 8-aligned, all other rows unaligned. */
static inline void win8(const double *restrict e, const double *restrict a,
                        const double *restrict b, const double *restrict c,
                        const double *restrict d, double *restrict o,
                        int64_t k, __m512d vc1, __m512d vc2)
{
    __m512d v0  = _mm512_load_pd(e + k);
    __m512d va  = _mm512_loadu_pd(a + k);
    __m512d vb  = _mm512_loadu_pd(b + k);
    __m512d vn1 = _mm512_permutex2var_pd(v0, _mm512_setr_epi64(0, 0, 1, 2, 3, 4, 5, 6), v0);
    __m512d vp1 = _mm512_permutex2var_pd(v0, _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 0), v0);
    vn1 = _mm512_mask_blend_pd(0x1, vn1, _mm512_set1_pd(e[k - 1]));
    vp1 = _mm512_mask_blend_pd(0x80, vp1, _mm512_set1_pd(e[k + 8]));
    __m512d r = _mm512_mul_pd(vc1, v0);
    r = _mm512_fmadd_pd(vc2, va, r);
    r = _mm512_fmadd_pd(vc2, vb, r);
    r = _mm512_fmadd_pd(vc2, _mm512_loadu_pd(c + k), r);
    r = _mm512_fmadd_pd(vc2, _mm512_loadu_pd(d + k), r);
    r = _mm512_fmadd_pd(vc2, vn1, r);
    r = _mm512_fmadd_pd(vc2, vp1, r);
    _mm512_storeu_pd(o + k, r);
}

/* Four adjacent slabs i0..i0+3, one row j. */
static inline void row4(const double *restrict In, double *restrict Out,
                        int64_t i0, int64_t j, int64_t N,
                        double c1, double c2,
                        __m512d vc1, __m512d vc2)
{
    (void)c1; (void)c2;
    const int64_t M = N - 2;
    const double *e0 = In + ((i0 + 0) * N + j) * N;
    const double *e1 = In + ((i0 + 1) * N + j) * N;
    const double *e2 = In + ((i0 + 2) * N + j) * N;
    const double *e3 = In + ((i0 + 3) * N + j) * N;
    double *o0 = Out + ((i0 + 0) * N + j) * N;
    double *o1 = Out + ((i0 + 1) * N + j) * N;
    double *o2 = Out + ((i0 + 2) * N + j) * N;
    double *o3 = Out + ((i0 + 3) * N + j) * N;
    const double *a0 = In + ((i0 - 1) * N + j) * N;
    const double *b3 = In + ((i0 + 4) * N + j) * N;

    const double *e[4]  = { e0, e1, e2, e3 };
    double       *o[4]  = { o0, o1, o2, o3 };
    const double *a[4]  = { a0, e0, e1, e2 };
    const double *b[4]  = { e1, e2, e3, b3 };
    const double *cp[4] = { e0 - N, e1 - N, e2 - N, e3 - N };
    const double *dp[4] = { e0 + N, e1 + N, e2 + N, e3 + N };

    int64_t k0[4], T[4];
    for (int s = 0; s < 4; ++s) {
        int64_t k = (8 - (((uintptr_t)(e[s]) >> 3) % 8)) % 8;
        if (k == 0) k = 8;
        k0[s] = k;
        T[s] = (k + 7 <= M) ? (M - k - 7) / 8 + 1 : 0;
    }
    int64_t Tmin = T[0];
    if (T[1] < Tmin) Tmin = T[1];
    if (T[2] < Tmin) Tmin = T[2];
    if (T[3] < Tmin) Tmin = T[3];

    /* scalar head */
    for (int s = 0; s < 4; ++s)
        for (int64_t k = 1; k < k0[s]; ++k)
            o[s][k] = dot4(a[s], b[s], cp[s], dp[s], e[s], k, c1, c2);

    /* vector main: one 8-window per slab */
    for (int64_t t = 0; t < Tmin; ++t) {
        win8(e[0], a[0], b[0], cp[0], dp[0], o[0], k0[0] + 8 * t, vc1, vc2);
        win8(e[1], a[1], b[1], cp[1], dp[1], o[1], k0[1] + 8 * t, vc1, vc2);
        win8(e[2], a[2], b[2], cp[2], dp[2], o[2], k0[2] + 8 * t, vc1, vc2);
        win8(e[3], a[3], b[3], cp[3], dp[3], o[3], k0[3] + 8 * t, vc1, vc2);
    }

    /* vector + scalar tail (per-slab extra windows if k0 differs) */
    for (int s = 0; s < 4; ++s) {
        int64_t k = k0[s] + 8 * Tmin;
        for (; k + 7 <= M; k += 8)
            win8(e[s], a[s], b[s], cp[s], dp[s], o[s], k, vc1, vc2);
        for (; k <= M; ++k)
            o[s][k] = dot4(a[s], b[s], cp[s], dp[s], e[s], k, c1, c2);
    }
}

static inline void slab(const double *restrict In, double *restrict Out,
                        int64_t i, int64_t N, double c1, double c2,
                        __m512d vc1, __m512d vc2)
{
    const int64_t M = N - 2;
    const double *ei = In + (i * N) * N;
    double *o0 = Out + (i * N) * N;

    for (int64_t j = 1; j <= M; ++j) {
        const double *a = In + ((i - 1) * N + j) * N;
        const double *b = In + ((i + 1) * N + j) * N;
        const double *c = ei + (j - 1) * N;
        const double *d = ei + (j + 1) * N;
        const double *e = ei + j * N;
        double *o = o0 + j * N;

        int64_t k = (8 - (((uintptr_t)(e) >> 3) % 8)) % 8;
        if (k == 0) k = 8;
        for (int64_t kk = 1; kk < k; ++kk)
            o[kk] = dot4(a, b, c, d, e, kk, c1, c2);
        for (; k + 7 <= M; k += 8)
            win8(e, a, b, c, d, o, k, vc1, vc2);
        for (; k <= M; ++k)
            o[k] = dot4(a, b, c, d, e, k, c1, c2);
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha)
{
    const int64_t M = N - 2;
    const int64_t G = M / 4;          /* full 4-slab groups */
    const double c1 = 1.0 - 6.0 * alpha;
    const double c2 = alpha;
    const __m512d vc1 = _mm512_set1_pd(c1);
    const __m512d vc2 = _mm512_set1_pd(c2);

    #pragma omp parallel
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp for schedule(dynamic, 2)
        for (int64_t g = 0; g < G; ++g)
            for (int64_t j = 1; j <= M; ++j)
                row4(A, B, 4 * g + 1, j, N, c1, c2, vc1, vc2);
        #pragma omp for schedule(dynamic, 4)
        for (int64_t i = 4 * G + 1; i <= M; ++i)
            slab(A, B, i, N, c1, c2, vc1, vc2);
        #pragma omp for schedule(dynamic, 2)
        for (int64_t g = 0; g < G; ++g)
            for (int64_t j = 1; j <= M; ++j)
                row4(B, A, 4 * g + 1, j, N, c1, c2, vc1, vc2);
        #pragma omp for schedule(dynamic, 4)
        for (int64_t i = 4 * G + 1; i <= M; ++i)
            slab(B, A, i, N, c1, c2, vc1, vc2);
    }
}
