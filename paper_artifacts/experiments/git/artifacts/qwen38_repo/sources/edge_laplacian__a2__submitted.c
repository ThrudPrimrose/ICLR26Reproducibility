/* Optimized weighted graph-Laplacian:
 *   flux[e] = w[e] * (x[src[e]] - x[dst[e]]);  Lx[src[e]] += flux;  Lx[dst[e]] -= flux;
 *
 * The naive reference does four sequential passes (zero, flux, scatter-src,
 * scatter-dst).  This version fuses everything into ONE OpenMP-parallel pass
 * that updates Lx in place with a compare-and-swap add: no flux scratch, no
 * re-reads of src/dst/w, and unrolled independent CAS chains keep the store
 * queue full.  A 2-edge-ahead prefetch of the Lx target lines hides the
 * cache-line fetch/upgrade latency of the locked operations.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static inline double __bitd(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static inline uint64_t __bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

/* f = old + v with a CAS loop (C has no hardware fp add-atomic).
 * On CAS failure the builtin updates *old with the current value, so no
 * extra load is needed before the retry. */
static inline void __fadd64(double *p, double v) {
    uint64_t old; memcpy(&old, p, 8);
    for (;;) {
        const uint64_t nv = __bits(__bitd(old) + v);
        if (__atomic_compare_exchange_n((uint64_t *)p, &old, nv, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst, const int64_t *restrict src,
                         const double *restrict w, const double *restrict x,
                         const int64_t E, const int64_t N) {
    if (N <= 0) return;
    if (E == 0) { memset(Lx, 0, (size_t)N * sizeof(double)); return; }

    int T = omp_get_max_threads();
    if (T < 1) T = 1;

    /* Tiny problems: thread start-up and cross-CCD line migration cost more than
     * the work; a single fused sequential pass wins. */
    if (E < 50000 || T < 2) {
        for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
        for (int64_t e = 0; e < E; ++e) {
            double f = w[e] * (x[src[e]] - x[dst[e]]);
            Lx[src[e]] += f;
            Lx[dst[e]] -= f;
        }
        return;
    }

    memset(Lx, 0, (size_t)N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int64_t e0 = 0; e0 < (E & ~(int64_t)3); e0 += 4) {
        const int64_t s0 = src[e0],     d0 = dst[e0];
        const int64_t s1 = src[e0 + 1], d1 = dst[e0 + 1];
        const int64_t s2 = src[e0 + 2], d2 = dst[e0 + 2];
        const int64_t s3 = src[e0 + 3], d3 = dst[e0 + 3];

        const int64_t a0 = e0 + 8, a1 = e0 + 9;
        if (a1 < E) {
            __builtin_prefetch(&Lx[src[a0]], 0, 0);
            __builtin_prefetch(&Lx[dst[a0]], 0, 0);
            __builtin_prefetch(&Lx[src[a1]], 0, 0);
            __builtin_prefetch(&Lx[dst[a1]], 0, 0);
        }

        const double f0 = w[e0]     * (x[s0] - x[d0]);
        const double f1 = w[e0 + 1] * (x[s1] - x[d1]);
        const double f2 = w[e0 + 2] * (x[s2] - x[d2]);
        const double f3 = w[e0 + 3] * (x[s3] - x[d3]);

        __fadd64(&Lx[s0], f0); __fadd64(&Lx[d0], -f0);
        __fadd64(&Lx[s1], f1); __fadd64(&Lx[d1], -f1);
        __fadd64(&Lx[s2], f2); __fadd64(&Lx[d2], -f2);
        __fadd64(&Lx[s3], f3); __fadd64(&Lx[d3], -f3);
    }
    for (int64_t e = E & ~(int64_t)3; e < E; ++e) {
        const int64_t s = src[e];
        const int64_t d = dst[e];
        const double f = w[e] * (x[s] - x[d]);
        __fadd64(&Lx[s], f);
        __fadd64(&Lx[d], -f);
    }
}
