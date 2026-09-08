#include <stdint.h>
#include <omp.h>

void segment_reduce_ragged_fp64(double *val, int64_t *row_ptr, double *w, double *out, int64_t NSEG, uint8_t *workspace, int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;
    for (int64_t s = 0; s < NSEG; ++s) {
        const int64_t start = row_ptr[s];
        const int64_t end = row_ptr[s + 1];
        double acc = 0.0;
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
