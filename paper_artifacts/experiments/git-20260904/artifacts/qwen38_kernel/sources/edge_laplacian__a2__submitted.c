#include <stdint.h>
#include <string.h>
#include <omp.h>

/* C-ABI v2 canonical order (contract.py Sec. 4: pointers sorted by name,
   scalars sorted by name):
     (Lx, dst, src, w, x, E, N, workspace, workspace_size)
   Semantics (numpy reference):
     Lx = 0; flux = w * (x[src] - x[dst]);
     Lx[src] += flux (add.at);  Lx[dst] -= flux (add.at).

   Parallel strategy: T per-thread private partial accumulators (workspace,
   T*N doubles); each thread sums its contiguous edge slice into its own
   copy (no atomics, no cross-thread traffic on the 178MB accumulator),
   then a streaming reduction overwrites Lx. */
void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst,
                         const int64_t *restrict src, const double *restrict w,
                         const double *restrict x, int64_t E, int64_t N,
                         void *workspace, int64_t workspace_size)
{
    if (E < (1 << 16)) {
        /* small instances (edge probes, preset S): plain serial */
        for (int64_t i = 0; i < N; i++) Lx[i] = 0.0;
        for (int64_t e = 0; e < E; e++) {
            int64_t s = src[e], d = dst[e];
            double flux = w[e] * (x[s] - x[d]);
            Lx[s] += flux;
            Lx[d] -= flux;
        }
        return;
    }
    if (workspace == NULL || workspace_size < (int64_t)192 * N) {
        /* no workspace: fall back to serial (still correct) */
        for (int64_t i = 0; i < N; i++) Lx[i] = 0.0;
        for (int64_t e = 0; e < E; e++) {
            int64_t s = src[e], d = dst[e];
            double flux = w[e] * (x[s] - x[d]);
            Lx[s] += flux;
            Lx[d] -= flux;
        }
        return;
    }

    int maxt = (int)omp_get_max_threads();
    int T = maxt > 24 ? 24 : maxt;   /* workspace sized for up to 24 partials */
    if (T < 1) T = 1;

    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        double *Lt = (double *)workspace + (size_t)t * (size_t)N;
        memset(Lt, 0, (size_t)N * sizeof(double));
        int64_t lo = (E * (int64_t)t) / T, hi = (E * (int64_t)(t + 1)) / T;
        for (int64_t e = lo; e < hi; e++) {
            int64_t s = src[e], d = dst[e];
            double flux = w[e] * (x[s] - x[d]);
            Lt[s] += flux;
            Lt[d] -= flux;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        double s = 0.0;
        for (int t = 0; t < T; t++)
            s += ((double *)workspace + (size_t)t * (size_t)N)[n];
        Lx[n] = s;
    }
}
