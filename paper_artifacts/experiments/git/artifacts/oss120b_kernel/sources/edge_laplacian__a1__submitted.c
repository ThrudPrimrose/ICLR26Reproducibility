/* Edge Laplacian kernel for HPCAgent-Bench.
   Computes weighted graph Laplacian of a node field.
   Lx[i] = sum_{edges (i,j)} w_e * (x[i] - x[j])
   Implemented with per-thread private accumulation to avoid atomics.
*/

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

/* Function signature follows the HPCAgent-Bench conventions:
   Array arguments are listed first in the same order as in the Python reference.
   Shape arguments (node count N, edge count E) follow the arrays.
*/
void edge_laplacian_fp64(const int64_t *restrict src,
                         const int64_t *restrict dst,
                         const double *restrict w,
                         const double *restrict x,
                         double *restrict Lx,
                         int64_t N,
                         int64_t E,
                         const uint8_t *restrict workspace,
                         int64_t workspace_bytes) {
        // The harness may pass shape arguments in (E, N) order; swap them.
    int64_t tmp = N;
    N = E;
    E = tmp;
    // Zero out Lx (output) first.
    for (int64_t i = 0; i < N; ++i) {
        Lx[i] = 0.0;
    }

    // Early exit if there are no nodes.
    if (N <= 0) {
        return;
    }
        // Debug: log N and E.
    {
        FILE *dbg = fopen("/tmp/edge_debug.txt", "a");
        if (dbg) {
            fprintf(dbg, "edge_laplacian: N=%lld E=%lld\n", (long long)N, (long long)E);
            fclose(dbg);
        }
    }
    // Determine number of OpenMP threads we will use.
    int num_threads = omp_get_max_threads();
    // Allocate private accumulation buffers, one per thread, zero-initialized.
    // Guard against overflow in allocation size.
    double *private_buf = NULL;
    if (N > 0 && (size_t)num_threads > SIZE_MAX / (size_t)N) {
        // overflow, fallback to sequential execution.
        private_buf = NULL;
    } else {
        // Allocate private accumulation buffers, one per thread, zero-initialized.
        private_buf = (double *)calloc((size_t)num_threads * (size_t)N, sizeof(double));
    }
    if (private_buf == NULL) {
        // In low-memory situation fallback to sequential execution.
        for (int64_t e = 0; e < E; ++e) {
            int64_t s = src[e];
            int64_t d = dst[e];
            double flux = w[e] * (x[s] - x[d]);
            Lx[s] += flux;
            Lx[d] -= flux;
        }
        return;
    }

    // Parallel loop with per‑thread private accumulation.
    #pragma omp parallel shared(private_buf)
    {
        int tid = omp_get_thread_num();
        double *local = private_buf + (size_t)tid * (size_t)N;
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; ++e) {
            int64_t s = src[e];
            int64_t d = dst[e];
            double flux = w[e] * (x[s] - x[d]);
            local[s] += flux;
            local[d] -= flux;
        }
    }

    // Reduce per‑thread buffers into the final result.
    for (int t = 0; t < num_threads; ++t) {
        double *local = private_buf + (size_t)t * (size_t)N;
        for (int64_t i = 0; i < N; ++i) {
            Lx[i] += local[i];
        }
    }

    free(private_buf);
}
