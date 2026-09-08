#include <stdint.h>
#include <omp.h>

void segment_reduce_ragged_fp64(double *restrict out, const int64_t *restrict row_ptr, const double *restrict val, const double *restrict w, const int64_t NSEG, uint8_t *restrict workspace, const int64_t workspace_bytes) {
        int64_t total = row_ptr[NSEG];
    #pragma omp target map(to: row_ptr[0:NSEG+1], val[0:total], w[0:total]) map(tofrom: out[0:NSEG])
    #pragma omp teams distribute parallel for schedule(static)
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        // Vectorize inner loop
        #pragma omp simd reduction(+:acc)
        for (int64_t e = row_ptr[s]; e < row_ptr[s + 1]; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
