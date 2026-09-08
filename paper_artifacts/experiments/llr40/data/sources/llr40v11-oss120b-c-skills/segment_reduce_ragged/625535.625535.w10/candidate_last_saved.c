#include <stdint.h>
#include <omp.h>

void segment_reduce_ragged_fp64(const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                double *restrict out,
                                const int64_t LEN,
                                const int64_t UNUSED) {
    int64_t NSEG = -1;
    const int64_t max_i = 10000000; // safeguard against infinite loop
    for (int64_t i = 0; i < max_i; ++i) {
        int64_t v = row_ptr[i];
        if (v == LEN) {
            // LEN matches total number of entries
            NSEG = i;
            break;
        }
        // If we have passed the possible segment count without a match,
        // we assume LEN is the number of segments and stop early.
        if (i > LEN && v > LEN) {
            break;
        }
    }
    if (NSEG == -1) {
        // LEN is likely the number of segments
        NSEG = LEN;
    }
    #pragma omp parallel for schedule(static)
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
