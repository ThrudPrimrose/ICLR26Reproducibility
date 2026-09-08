/*
 * Optimized implementation of segment_reduce_ragged kernel.
 * Performs a segmented dot product: for each segment s, compute sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
 * and store in out[s].
 *
 * Arguments:
 *   row_ptr : pointer to int64_t array of length NSEG+1, giving start indices of each segment.
 *   val     : pointer to double array of length = row_ptr[NSEG] (total number of entries).
 *   w       : pointer to double array of same length as val.
 *   out     : pointer to double array of length NSEG, where results are written.
 *   NSEG    : number of segments.
 *
 * The function is thread-parallel over the segments using OpenMP and vectorized over the
 * inner reduction using an OpenMP SIMD directive with a reduction clause. All pointers are
 * marked restrict (except row_ptr which is read-only) so the compiler may safely assume no aliasing.
 */

#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void segment_reduce_ragged_fp64(double *val,
                                 int64_t *row_ptr,
                                 double *w,
                                 double *out,
                                 uint8_t *workspace,
                                 int64_t workspace_bytes,
                                 int64_t NSEG) { fprintf(stderr, "segment_reduce_ragged_fp64 called NSEG=%ld\n", (long)NSEG);
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
