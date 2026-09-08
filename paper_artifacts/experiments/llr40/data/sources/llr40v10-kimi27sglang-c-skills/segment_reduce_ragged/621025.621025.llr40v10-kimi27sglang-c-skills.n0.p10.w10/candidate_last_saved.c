#include <stdint.h>
#include <omp.h>

void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG,
                                uint8_t *restrict workspace,
                                const int64_t workspace_bytes)
{
#pragma omp parallel for schedule(guided)
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        const int64_t e0 = row_ptr[s];
        const int64_t e1 = row_ptr[s + 1];
#pragma omp simd reduction(+:acc)
        for (int64_t e = e0; e < e1; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
