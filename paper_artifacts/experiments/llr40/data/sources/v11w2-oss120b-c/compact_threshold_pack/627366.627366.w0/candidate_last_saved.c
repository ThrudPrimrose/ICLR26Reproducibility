#include <stdint.h>

// Stream compaction: pack src[i] * weight[i] for each src[i] > 0.
// The output count is stored in out_count[0].
// All arrays are of length LEN_1D.

void compact_threshold_pack(const double *restrict src,
                           const double *restrict weight,
                           double *restrict packed,
                           int64_t *restrict out_count,
                           const int64_t LEN_1D) {
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double s = src[i];
        if (s > 0.0) {
            packed[n] = s * weight[i];
            n++;
        }
    }
    out_count[0] = n;
}
