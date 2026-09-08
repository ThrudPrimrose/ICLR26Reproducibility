#include <stdint.h>

void segment_reduce_ragged_fp64(int64_t *row_ptr, const double *val, const double *w, double *out, int64_t NSEG) {
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        for (int64_t e = row_ptr[s]; e < row_ptr[s+1]; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
