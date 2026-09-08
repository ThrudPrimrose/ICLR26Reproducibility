#include <stdint.h>
#include <omp.h>

/* versioned_distance_update kernel.
 * Computes a[i] = 0.75 * a[i - K] + b[i] * c[i] for i = K .. LEN_1D-1.
 * The arrays a, b, c are of length LEN_1D.
 * K is a runtime distance, may be 1 (recurrence), small, or large.
 * For K == 1 the loop is a strict recurrence and must be executed serially.
 * For K > 1 we have K independent chains; we parallelise across chains.
 */

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D) {
    // No work needed if K is out of range.
    if (K <= 0 || K >= LEN_1D) {
        return;
    }

    // Special case K == 1: pure recurrence, keep serial.
    if (K == 1) {
        // Unrolled loop for recurrence to improve performance.
        int64_t i = 1;
        const int64_t limit = LEN_1D - 7;
        for (; i < limit; i += 8) {
            a[i] = 0.75 * a[i - 1] + b[i] * c[i];
            a[i + 1] = 0.75 * a[i] + b[i + 1] * c[i + 1];
            a[i + 2] = 0.75 * a[i + 1] + b[i + 2] * c[i + 2];
            a[i + 3] = 0.75 * a[i + 2] + b[i + 3] * c[i + 3];
            a[i + 4] = 0.75 * a[i + 3] + b[i + 4] * c[i + 4];
            a[i + 5] = 0.75 * a[i + 4] + b[i + 5] * c[i + 5];
            a[i + 6] = 0.75 * a[i + 5] + b[i + 6] * c[i + 6];
            a[i + 7] = 0.75 * a[i + 6] + b[i + 7] * c[i + 7];
        }
        for (; i < LEN_1D; ++i) {
            a[i] = 0.75 * a[i - 1] + b[i] * c[i];
        }
        return;
    }

    // K > 1: K independent chains. Parallelise over the chain offset.
    #pragma omp parallel for schedule(static) default(none) \
        shared(a, b, c, LEN_1D, K)
    for (int64_t offset = 0; offset < K; ++offset) {
        // Process the chain starting at index offset + K.
        for (int64_t i = offset + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
