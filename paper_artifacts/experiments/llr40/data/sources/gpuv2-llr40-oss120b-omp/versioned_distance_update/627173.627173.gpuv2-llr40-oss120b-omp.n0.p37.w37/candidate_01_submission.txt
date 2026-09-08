#include <stdint.h>

void versioned_distance_update_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                                   const int64_t K, const int64_t LEN_1D) {
    if (K <= 0) return;
    // For very small arrays, use a simple serial implementation to avoid OpenMP overhead.
    if (LEN_1D <= 1024) {
        for (int64_t i = K; i < LEN_1D; ++i) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
        return;
    }
    const int64_t CHUNK = 8000000; // chunk size (elements) for larger inputs
    for (int64_t chunk_start = 0; chunk_start < LEN_1D; chunk_start += CHUNK) {
        int64_t seg_start = (chunk_start >= K) ? (chunk_start - K) : 0;
        int64_t seg_end = chunk_start + CHUNK;
        if (seg_end > LEN_1D) seg_end = LEN_1D;
        int64_t seg_len = seg_end - seg_start;
        #pragma omp target map(to: b[seg_start:seg_len], c[seg_start:seg_len]) map(tofrom: a[seg_start:seg_len])
        {
            #pragma omp teams distribute parallel for schedule(static)
            for (int64_t r = 0; r < K; ++r) {
                int64_t base = seg_start + K;
                int64_t i_start = base + ((r - (base % K) + K) % K);
                for (int64_t i = i_start; i < seg_end; i += K) {
                    a[i] = 0.75 * a[i - K] + b[i] * c[i];
                }
            }
        }
    }
}
