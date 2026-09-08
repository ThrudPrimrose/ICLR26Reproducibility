#include <stdint.h>
#pragma STDC FP_CONTRACT OFF


void segment_reduce_ragged_fp64(const int64_t * restrict row_ptr,
                                const double * restrict val,
                                const double * restrict w,
                                int64_t NSEG,
                                double * restrict out,
                                uint8_t * restrict workspace,
                                const int64_t workspace_bytes) {
        for (int64_t s = 0; s < NSEG; ++s) {
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        int64_t len = end - start;
        const double * v = val + start;
        const double * restrict ww = w + start;
        double acc = 0.0;
        for (int64_t i = 0; i < len; ++i) {
            volatile double prod = v[i] * ww[i];
            acc += prod;
        }
        out[s] = acc;
    }
}

