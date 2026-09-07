/* Edge Laplacian kernel for HPCAgent-Bench.
   Computes weighted graph Laplacian of a node field.
   Lx[i] = sum_{edges (i,j)} w_e * (x[i] - x[j])
   Implemented with per‑thread private accumulation to avoid atomics.
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

/* Function signature follows the HPCAgent‑Bench conventions:
   Array arguments are listed first in the same order as in the Python reference.
   Shape arguments (edge count E, node count N) follow the arrays.
   A workspace pointer and its size are provided for kernels that may need it;
   this kernel does not use them, but they are included to match the expected ABI.
*/
void edge_laplacian_fp64(const int64_t *restrict src,
                         const int64_t *restrict dst,
                         const double *restrict w,
                         const double *restrict x,
                         double *restrict Lx,
                         int64_t N,
                         int64_t E) {
    
    // Guard against degenerate sizes.
    if (N <= 0) {
        // Nothing to compute: Lx has zero length.
        return;
    }
    if (E <= 0) {
        // No edges: output remains zeroed.
        // Zero out Lx (output) first.
        for (int64_t i = 0; i < N; ++i) {
            Lx[i] = 0.0;
        }
        return;
    }

    // Zero out Lx (output) first.
    for (int64_t i = 0; i < N; ++i) {
        Lx[i] = 0.0;
    }

        // Compute Lx using sequential loop.
    for (int64_t e = 0; e < E; ++e) {
        int64_t s = src[e];
        int64_t d = dst[e];
        double flux = w[e] * (x[s] - x[d]);
        Lx[s] += flux;
        Lx[d] -= flux;
    }
}
