#include <stdint.h>
void compact_threshold_pack_fp64(const double *src,
                                 const double *weight,
                                 double *packed,
                                 int64_t *out_count,
                                 const int64_t LEN_1D) {
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D && n < 1000000; ++i) {
        double s = src[i];
        if (s > 0.0) {
            packed[n++] = s * weight[i];
        }
    }
    out_count[0] = n;
}
