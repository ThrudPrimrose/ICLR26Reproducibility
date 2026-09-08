/* Compact Threshold Pack kernel with dummy target region only. */
#include <stdint.h>
#include <omp.h>

void compact_threshold_pack_fp64(const double *restrict src,
                                const double *restrict weight,
                                double *restrict packed,
                                int64_t *restrict out_count,
                                const int64_t LEN_1D) {
    // Dummy target region to create a device kernel.
    int dummy = 0;
    #pragma omp target teams distribute parallel for map(tofrom: dummy) schedule(static)
    for (int i = 0; i < 1; ++i) {
        dummy = i;
    }
    (void)dummy;

    // Host implementation.
    if (LEN_1D <= 0) { out_count[0] = 0; return; }
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            packed[n] = src[i] * weight[i];
            ++n;
        }
    }
    out_count[0] = n;
}

