/*
 * versioned_distance_update kernel implementation in C.
 * Updates array a in-place: a[i] = 0.75 * a[i-K] + b[i] * c[i] for i = K .. LEN_1D-1.
 * Supports runtime distance K; parallelizes across independent chains.
 */

#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a,
                                   const double *restrict b,
                                   const double *restrict c,
                                   const int64_t LEN_1D,
                                   const int64_t K) {
    // For indices i < K, a[i] remains unchanged (as per reference).
    // Parallelize over K independent chains (offsets).
    #pragma omp parallel for schedule(static)
    for (int64_t offset = 0; offset < K; ++offset) {
        // First index in this chain that is updated is offset + K.
        // If offset + K >= LEN_1D, the loop does nothing.
                #pragma omp simd
        for (int64_t i = offset + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
