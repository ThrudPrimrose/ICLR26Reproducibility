#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t K, const int64_t LEN_1D) {
    if (LEN_1D <= K) return;
    // Parallel across K independent chains.
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < K; ++r) {
        // Process chain starting at index r + K.
        for (int64_t i = r + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
