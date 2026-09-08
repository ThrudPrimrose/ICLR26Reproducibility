/*
 * compact_threshold_pack: serial implementation with unused workspace arguments.
 */

#include <stdint.h>
#include <stddef.h>

void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 double *restrict packed,
                                 int64_t *restrict out_count,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    int64_t n = 0;
    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            packed[n] = src[i] * weight[i];
            ++n;
        }
    }
    out_count[0] = n;
}
