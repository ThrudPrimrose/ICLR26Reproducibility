// Weighted graph-Laplacian over an edge list:
//   Lx[v] = sum over edges (s,d) of w*(x[s]-x[d]) * (v==s ? +1 : v==d ? -1 : 0)
// Per-thread partial accumulation buffers (zeroed by the harness per rep,
// untimed), then a parallel combine that rebuilds Lx from scratch.
#include <stdint.h>
#include <string.h>
#include <omp.h>

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst,
                         const int64_t *restrict src, const double *restrict w,
                         const double *restrict x, const int64_t E, const int64_t N,
                         uint8_t *restrict workspace, int64_t workspace_size) {
    if (E <= 0 || N <= 0) return;

    const int NT = 16;
    int maxt = omp_get_max_threads();
    int nt = maxt < NT ? maxt : NT;
    if (nt < 1) nt = 1;

    if (workspace && workspace_size >= (int64_t)nt * ((int64_t)N * 8 + 64)) {
        const int64_t stride = N + 8; /* doubles; +8 = 64B pad keeps buffers aligned */
        double *Ptab[NT];
        for (int t = 0; t < nt; ++t) Ptab[t] = (double *)workspace + (size_t)t * (size_t)stride;

        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            double *P = Ptab[tid];
            const int64_t chunk = (E + nt - 1) / nt;
            int64_t b0 = (int64_t)tid * chunk;
            int64_t b1 = b0 + chunk;
            if (b1 > E) b1 = E;
            for (int64_t i = b0; i < b1; ++i) {
                const int64_t s = src[i];
                const int64_t d = dst[i];
                const double f = w[i] * (x[s] - x[d]);
                P[s] += f;
                P[d] -= f;
            }
        }

        #pragma omp parallel for num_threads(nt) schedule(static)
        for (int64_t v = 0; v < N; ++v) {
            double acc = 0.0;
            for (int t = 0; t < nt; ++t) acc += Ptab[t][v];
            Lx[v] = acc;
        }
        return;
    }

    /* Fallback without scratch: Lx is rebuilt by zero + scatter (no reliance on
       entry state). */
    #pragma omp parallel for schedule(static)
    for (int64_t v = 0; v < N; ++v) Lx[v] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < E; ++i) {
        const int64_t s = src[i];
        const int64_t d = dst[i];
        const double f = w[i] * (x[s] - x[d]);
        #pragma omp atomic update
        Lx[s] += f;
        #pragma omp atomic update
        Lx[d] -= f;
    }
}
