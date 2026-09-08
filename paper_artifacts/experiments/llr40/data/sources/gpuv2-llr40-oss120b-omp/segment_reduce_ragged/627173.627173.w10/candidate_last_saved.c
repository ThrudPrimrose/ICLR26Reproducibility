#include <stdint.h>

void segment_reduce_ragged_fp64(const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                double *restrict out,
                                const int64_t NSEG) {
    #pragma omp target parallel for schedule(static) map(to: row_ptr[0:NSEG+1]) map(from: out[0:NSEG])
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        #pragma omp simd reduction(+:acc)
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
