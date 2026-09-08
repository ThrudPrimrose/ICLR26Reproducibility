/*
 * Optimized implementation of the scatter_accum_dup kernel.
 * Performs indexed accumulation: bins[ip[i]] += src[i] for i in [0, LEN_1D).
 * Handles potential index collisions using OpenMP atomics.
 *
 * The kernel is expected to be linked as a shared library with the symbol
 *   scatter_accum_dup
 * and called with the signature:
 *   void scatter_accum_dup(double *restrict bins,
 *                               const double *restrict src,
 *                               const int32_t *restrict ip,
 *                               const int64_t LEN_1D);
 */

#include <stdint.h>
#include <stddef.h>

/* Use OpenMP for parallelism. */
void scatter_accum_dup_fp64(double *restrict bins,
                            const int32_t *restrict ip,
                            const double *restrict src,
                            const int64_t LEN_1D,
                            uint8_t *workspace,
                            const int64_t workspace_len) {
    /* Parallel loop with OpenMP. The reduction is performed via an atomic
     * addition because the indices in `ip` may repeat, leading to true
     * reductions that need synchronization.
     */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int32_t idx = ip[i];
        double val = src[i];
        #pragma omp atomic
        bins[idx] += val;
    }
}

