// Optimized weighted graph-Laplacian scatter-add.
// Lx[v] = sum over edges of flux contributions (src +f, dst -f), where
// f = w[i] * (x[src[i]] - x[dst[i]]).
//
// Strategy: per-thread partial-sum buffers (NUMA-local via in-thread
// malloc), a fused flux+scatter pass with deep software prefetch, then a
// parallel combine. A plain fallback covers degenerate sizes.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void edge_laplacian_fallback(double *restrict Lx, const int64_t *restrict dst,
                                    const int64_t *restrict src, const double *restrict w,
                                    const double *restrict x, const int64_t E, const int64_t N) {
    double *flux = (double*)malloc((size_t)E * 8);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < E; ++i) flux[i] = w[i] * (x[src[i]] - x[dst[i]]);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < E; ++i) {
        double f = flux[i];
        #pragma omp atomic
        Lx[src[i]] += f;
        #pragma omp atomic
        Lx[dst[i]] -= f;
    }
    free(flux);
}

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst, const int64_t *restrict src, const double *restrict w, const double *restrict x, const int64_t E, const int64_t N) {
    if (N <= 0) return;
    if (E <= 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
        return;
    }
    int maxt = omp_get_max_threads();
    int T = maxt < 32 ? maxt : 32;
    if ((uint64_t)T * (uint64_t)N * 8U > 24ULL*1024*1024*1024) {
        T = (int)(24ULL*1024*1024*1024 / ((uint64_t)N * 8U));
        if (T < 1) T = 1;
    }
    if (T < 2) {
        edge_laplacian_fallback(Lx, dst, src, w, x, E, N);
        return;
    }
    double **buf = (double**)malloc((size_t)T * sizeof(double*));
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        double *b = (double*)malloc((size_t)N * sizeof(double));
        memset(b, 0, (size_t)N * sizeof(double));
        buf[t] = b;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < E; ++i) {
            int64_t s = src[i], d = dst[i];
            if (i + 256 < E) {
                __builtin_prefetch(&x[src[i+256]], 0, 1);
                __builtin_prefetch(&x[dst[i+256]], 0, 1);
                __builtin_prefetch(&b[src[i+256]], 1, 1);
                __builtin_prefetch(&b[dst[i+256]], 1, 1);
            }
            double f = w[i] * (x[s] - x[d]);
            b[s] += f;
            b[d] -= f;
        }
        #pragma omp for schedule(static)
        for (int64_t v = 0; v < N; ++v) {
            double s = 0.0;
            for (int u = 0; u < T; ++u) s += buf[u][v];
            Lx[v] = s;
        }
        free(b);
    }
    free(buf);
}
