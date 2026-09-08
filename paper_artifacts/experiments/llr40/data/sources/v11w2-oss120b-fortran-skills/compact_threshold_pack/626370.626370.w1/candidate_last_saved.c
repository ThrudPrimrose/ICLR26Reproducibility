#include <stddef.h>
#include <stdint.h>

void compact_threshold_pack_fp64(int64_t out_count[1], const double *src, const double *weight,
                                 double *packed, int64_t len_1d, void *workspace, int64_t workspace_size) {
    int64_t n = 0;
    for (int64_t i = 0; i < len_1d; ++i) {
        if (src[i] > 0.0) {
            packed[n] = src[i] * weight[i];
            ++n;
        }
    }
    out_count[0] = n;
}
