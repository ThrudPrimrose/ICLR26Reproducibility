/* ext_break_capture kernel (FP64) */
/*
 * The harness expects the following signature (order matters):
 *   void ext_break_capture_fp64(double *restrict a,
 *                               int64_t *restrict out_index,
 *                               double *restrict out_value,
 *                               int64_t K,
 *                               int64_t LEN_1D,
 *                               uint8_t *restrict workspace,
 *                               int64_t workspace_bytes);
 *
 * The kernel scans the array `a` for the first element greater than the
 * threshold `K`. The index and value of that element are written to the output
 * scalars `out_index[0]` and `out_value[0]`. If no element exceeds `K` the
 * outputs are set to the sentinel values -1 and -1.0.
 *
 * A parallel implementation is used to improve performance on large arrays.
 * Each thread scans a disjoint chunk, records the first qualifying index in its
 * chunk, and a critical section reduces to the globally smallest index.
 */

#include <stdint.h>
#include <omp.h>

void ext_break_capture_fp64(double *restrict a,
                             int64_t *restrict out_index,
                             double *restrict out_value,
                             int64_t K,
                             int64_t LEN_1D,
                             uint8_t *restrict workspace,
                             int64_t workspace_bytes)
{
    // Workspace arguments are unused for this kernel.
    (void)workspace;
    (void)workspace_bytes;

    // Initialise outputs to sentinel values.
    out_index[0] = -1;
    out_value[0] = -1.0;

    if (LEN_1D <= 0) return;

    // Shared variables for the best (smallest) index and its associated value.
    int64_t best_idx = LEN_1D;   // sentinel beyond any valid index
    double best_val = -1.0;
    double Kd = (double)K;      // Threshold as double for comparison

    // Parallel scan: each thread works on a contiguous chunk.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Compute chunk boundaries (static, contiguous).
        int64_t chunk = (LEN_1D + nthreads - 1) / nthreads; // ceil division
        int64_t i_start = tid * chunk;
        int64_t i_end = i_start + chunk;
        if (i_end > LEN_1D) i_end = LEN_1D;

        int64_t local_idx = LEN_1D; // sentinel meaning "no hit" in this chunk
        double local_val = -1.0;
        for (int64_t i = i_start; i < i_end; ++i) {
            if (a[i] > Kd) {
                local_idx = i;
                local_val = a[i];
                break; // stop scanning this chunk after first hit
            }
        }
        // Reduce to the global best index/value.
        #pragma omp critical
        {
            if (local_idx < best_idx) {
                best_idx = local_idx;
                best_val = local_val;
            }
        }
    }

    // Write the result if a qualifying element was found.
    if (best_idx < LEN_1D) {
        out_index[0] = best_idx;
        out_value[0] = best_val;
    }
    // Otherwise the sentinel values remain.
}
