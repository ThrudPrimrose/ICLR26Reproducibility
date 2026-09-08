#include <stdint.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Simple gather with prefetching to improve memory latency
    const int64_t PREFETCH_DIST = 64; // prefetch distance (adjustable)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        // Prefetch future data if within bounds
        if (i + PREFETCH_DIST < LEN_1D) {
            // Prefetch the target element from b and the index from ip
            __builtin_prefetch(&b[ip[i + PREFETCH_DIST]], 0, 1);
            __builtin_prefetch(&ip[i + PREFETCH_DIST], 0, 1);
        }
        a[i] = b[ip[i]];
    }
}
