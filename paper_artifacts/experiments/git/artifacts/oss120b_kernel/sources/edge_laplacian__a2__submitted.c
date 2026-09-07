#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>

/* Weighted graph Laplacian kernel.
 * Signature: src, dst, w, x, Lx, N (nodes), E (edges).
 */

void edge_laplacian_fp64(const int64_t *restrict src,
                         const int64_t *restrict dst,
                         const double *restrict w,
                         const double *restrict x,
                         double *restrict Lx,
                         int64_t N,
                         int64_t E)
{
    // Debug shape values and first few data.
    printf("DEBUG N=%ld E=%ld\n", (long)N, (long)E);
    fflush(stdout);
    // Print first few w and x values.
    int64_t dbg_len = (E < 5 ? E : 5);
    for (int64_t i = 0; i < dbg_len; ++i) {
        int64_t s = src[i];
        int64_t d = dst[i];
        printf("DEBUG data i=%ld w=%a x_s=%a x_d=%a\n", (long)i, ((float *)w)[i], x[s], x[d]);
    }

    // Zero the output array.
    memset(Lx, 0, (size_t)N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        int64_t s = src[e];
        int64_t d = dst[e];
        float w_val = ((float *)w)[e];
        double flux = (double)w_val * (x[s] - x[d]);
        #pragma omp atomic
        Lx[s] += flux;
        #pragma omp atomic
        Lx[d] -= flux;
    }
}
