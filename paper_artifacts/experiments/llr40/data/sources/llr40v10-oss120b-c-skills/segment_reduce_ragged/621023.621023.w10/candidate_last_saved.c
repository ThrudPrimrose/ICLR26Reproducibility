#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Segmented dot product over a ragged CSR-style structure.
 * For each segment s (0 <= s < NSEG), computes:
 *   out[s] = sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
 *
 * Parameters:
 *   row_ptr : pointer to int64_t array of length NSEG+1, defining segment boundaries.
 *   val     : pointer to double array of length row_ptr[NSEG] (total number of entries).
 *   w       : pointer to double array of same length as val.
 *   out     : pointer to double array of length NSEG (output).
 *   NSEG    : number of segments.
 */

void segment_reduce_ragged_fp64(double *restrict out,
                                 const int64_t *restrict row_ptr,
                                 const double *restrict val,
                                 const double *restrict w,
                                 const int64_t NSEG,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_bytes) {
    // Compute total entries.
    int64_t total = row_ptr[NSEG];
    // Parallel loop over segments.
        for (int64_t s = 0; s < NSEG; ++s) {
        double sum = 0.0;
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        if (end > total) end = total;
        #pragma omp simd reduction(+:sum)
        for (int64_t e = start; e < end; ++e) {
            sum += val[e] * w[e];
        }
        out[s] = sum;
    }
}
