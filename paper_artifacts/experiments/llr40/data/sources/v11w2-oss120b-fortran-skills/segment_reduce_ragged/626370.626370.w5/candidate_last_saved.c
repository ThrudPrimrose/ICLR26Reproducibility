#include <stddef.h>
#include <stdint.h>

void segment_reduce_ragged_fp64(const int64_t *row_ptr,
                               const double *val,
                               const double *w,
                               double *out,
                               int64_t NSEG,
                               const void *workspace,
                               int64_t workspace_size) {
    for (int64_t s = 0; s < NSEG; ++s) {
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s+1];
        double acc = 0.0;
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
    // workspace arguments are unused.
}
