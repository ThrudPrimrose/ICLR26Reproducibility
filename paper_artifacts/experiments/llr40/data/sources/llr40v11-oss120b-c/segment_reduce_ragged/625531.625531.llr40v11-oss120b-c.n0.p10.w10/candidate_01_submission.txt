/* Optimized C implementation for the segment_reduce_ragged kernel.
 * Computes a segmented dot product: for each segment s, out[s] = sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
 *
 * The function follows the v2 C-ABI naming convention used by the benchmark suite.
 * It operates on double-precision floating point data (fp64) and uses OpenMP to
 * parallelise the outer segment loop. The inner reduction is left to the compiler
 * for vectorisation. A guided schedule is chosen to amortise the load imbalance
 * arising from ragged segment lengths.
 */

#include <stdint.h>

/*
 * row_ptr: length NSEG+1, int64 indices marking the start of each segment within val and w.
 * val:     values array of total length (row_ptr[NSEG]), double precision.
 * w:       weights array of same length as val.
 * out:     output array of length NSEG.
 * NSEG:    number of segments.
 */
void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG) {
    // Parallelise over segments. Use a guided schedule to better handle variable segment lengths.
    #pragma omp parallel for schedule(guided)
    for (int64_t s = 0; s < NSEG; ++s) {
        int64_t start = row_ptr[s];
        int64_t end   = row_ptr[s + 1];
        double acc = 0.0;
        // Simple accumulator loop; rely on compiler vectorisation.
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
