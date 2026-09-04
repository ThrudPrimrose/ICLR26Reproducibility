#include <stdint.h>
#include <stdio.h>
#include <omp.h>

// Segmented reduction: compute dot product per segment
void segment_reduce_ragged_fp64(const double *restrict val,
                                const int64_t *restrict row_ptr,
                                const double *restrict w,
                                double *restrict out,
                                const int64_t NSEG,
                                uint8_t *restrict workspace,
                                const int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        for (int64_t e = row_ptr[s]; e < row_ptr[s + 1]; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
