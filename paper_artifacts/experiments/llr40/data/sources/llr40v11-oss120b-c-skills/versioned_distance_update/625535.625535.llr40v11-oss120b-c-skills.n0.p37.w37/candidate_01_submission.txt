#include <stdint.h>
#include <omp.h>

/*
 * versioned_distance_update kernel.
 * Implements the recurrence a[i] = 0.75 * a[i-K] + b[i] * c[i] for i = K .. LEN_1D-1.
 * The dependence distance K creates K independent chains (i % K is constant within a chain).
 * We parallelise over the K chains and keep the previous value of each chain in a register
 * to avoid loading a[i-K] from memory on every iteration.
 */
void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D) {
    if (K <= 0 || K >= LEN_1D) {
        return;
    }

    // Parallel over the K independent chains.
    #pragma omp parallel for schedule(static) default(none) \
        shared(a, b, c, K, LEN_1D)
    for (int64_t offset = 0; offset < K; ++offset) {
        // If the starting index is already beyond the array, skip.
        if (offset >= LEN_1D) {
            continue;
        }
        // Seed value for this chain: a[offset] (indices < K are not updated).
        double prev = a[offset];
        // Process the chain: i = offset + m*K, m >= 1.
        for (int64_t i = offset + K; i < LEN_1D; i += K) {
            double cur = 0.75 * prev + b[i] * c[i];
            a[i] = cur;
            prev = cur;
        }
    }
}
