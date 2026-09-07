// Weighted graph-Laplacian assembly over an edge list:
//   Lx = 0;  for e: f = w[e]*(x[src[e]] - x[dst[e]]);  Lx[src[e]] += f;  Lx[dst[e]] -= f;
//
// Strategies:
//  - small E: two sequential passes (one per endpoint), no temp array.
//  - large E: one fused pass reading src/dst/w exactly once, per-thread partial
//    buffers (no cross-thread RMW, no atomics), then a per-vertex combine.
//    Partial sums are combined in thread order; within a thread contributions
//    are applied in edge order, so the result matches the reference up to
//    floating-point reassociation (~1e-15 abs).
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

// bit-exact fallback: flux array (parallel fill, sequential in-order scatters)
static void seq_path(double *restrict Lx, const int64_t *restrict dst, const int64_t *restrict src,
                     const double *restrict w, const double *restrict x, int64_t E, int64_t N) {
    double *flux = (double *)malloc((size_t)E * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int64_t v = 0; v < N; ++v) Lx[v] = 0.0;
    if (flux) {
        #pragma omp parallel for schedule(static)
        for (int64_t e = 0; e < E; ++e) flux[e] = w[e] * (x[src[e]] - x[dst[e]]);
        for (int64_t e = 0; e < E; ++e) Lx[src[e]] += flux[e];
        for (int64_t e = 0; e < E; ++e) Lx[dst[e]] -= flux[e];
        free(flux);
    } else {
        for (int64_t e = 0; e < E; ++e) Lx[src[e]] += w[e] * (x[src[e]] - x[dst[e]]);
        for (int64_t e = 0; e < E; ++e) Lx[dst[e]] -= w[e] * (x[src[e]] - x[dst[e]]);
    }
}

#define UNROLL 8

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst, const int64_t *restrict src,
                         const double *restrict w, const double *restrict x, const int64_t E, const int64_t N) {
    if (N <= 0) return;
    if (E <= 0) {
        for (int64_t v = 0; v < N; ++v) Lx[v] = 0.0;
        return;
    }
    const int nt = omp_get_max_threads();
    // partial-buffer budget: nt*N doubles must stay affordable (<= 16 GiB)
    if (E < (1LL << 15) || nt < 2 || (uint64_t)nt * (uint64_t)N > (1uLL << 31)) {
        seq_path(Lx, dst, src, w, x, E, N);
        return;
    }
    static double *P = NULL;
    static int64_t capN = 0;
    static int capT = 0;
    if (N > capN || nt > capT) {
        free(P);
        int64_t wantN = N > capN ? N : capN;
        int wantT = nt > capT ? nt : capT;
        P = (double *)malloc((size_t)wantT * (size_t)wantN * sizeof(double));
        capN = capT = 0;
        if (P) { capN = wantN; capT = wantT; }
    }
    if (!P) { seq_path(Lx, dst, src, w, x, E, N); return; }

    #pragma omp parallel num_threads(nt)
    {
        const int t = omp_get_thread_num();
        double *Pt = P + (size_t)t * N;
        {
            int64_t v = 0;
            const int64_t n4 = N - (N & 3);
            for (; v < n4; v += 4) {
                Pt[v] = 0.0; Pt[v+1] = 0.0; Pt[v+2] = 0.0; Pt[v+3] = 0.0;
            }
            for (; v < N; ++v) Pt[v] = 0.0;
        }
        const int64_t i0 = (E * t) / nt, i1 = (E * (t + 1)) / nt;
        {
            int64_t e = i0;
            const int64_t e8 = i0 + (((i1 - i0) >> 3) << 3);
            for (; e < e8; e += UNROLL) {
                int64_t s0 = src[e], s1 = src[e+1], s2 = src[e+2], s3 = src[e+3],
                       s4 = src[e+4], s5 = src[e+5], s6 = src[e+6], s7 = src[e+7];
                int64_t d0 = dst[e], d1 = dst[e+1], d2 = dst[e+2], d3 = dst[e+3],
                       d4 = dst[e+4], d5 = dst[e+5], d6 = dst[e+6], d7 = dst[e+7];
                double f0 = w[e]   * (x[s0] - x[d0]);
                double f1 = w[e+1] * (x[s1] - x[d1]);
                double f2 = w[e+2] * (x[s2] - x[d2]);
                double f3 = w[e+3] * (x[s3] - x[d3]);
                double f4 = w[e+4] * (x[s4] - x[d4]);
                double f5 = w[e+5] * (x[s5] - x[d5]);
                double f6 = w[e+6] * (x[s6] - x[d6]);
                double f7 = w[e+7] * (x[s7] - x[d7]);
                Pt[s0] += f0; Pt[s1] += f1; Pt[s2] += f2; Pt[s3] += f3;
                Pt[s4] += f4; Pt[s5] += f5; Pt[s6] += f6; Pt[s7] += f7;
                Pt[d0] -= f0; Pt[d1] -= f1; Pt[d2] -= f2; Pt[d3] -= f3;
                Pt[d4] -= f4; Pt[d5] -= f5; Pt[d6] -= f6; Pt[d7] -= f7;
            }
            for (; e < i1; ++e) {
                int64_t s = src[e], d = dst[e];
                double f = w[e] * (x[s] - x[d]);
                Pt[s] += f;
                Pt[d] -= f;
            }
        }
    }
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t v = 0; v < N; ++v) {
        double acc = 0.0;
        for (int t = 0; t < nt; ++t) acc += P[(size_t)t * N + v];
        Lx[v] = acc;
    }
}
