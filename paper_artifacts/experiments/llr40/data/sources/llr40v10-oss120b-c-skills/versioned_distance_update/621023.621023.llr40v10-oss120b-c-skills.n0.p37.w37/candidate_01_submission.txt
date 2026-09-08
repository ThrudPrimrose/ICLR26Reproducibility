#include <stdint.h>
#include <omp.h>

/*
 * versioned_distance_update kernel.
 * Implements a runtime-distance recurrence:
 *   a[i] = 0.75 * a[i - K] + b[i] * c[i]
 * for i = K .. LEN_1D-1.
 * The first K elements of a are treated as seeds and left untouched.
 *
 * The kernel processes K independent chains in parallel using OpenMP.
 * For K == 1 the outer parallel loop has a single iteration,
 * and the `if` clause disables threading to avoid parallel overhead.
 */

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D) {
    if (K <= 0) return;
    #pragma omp parallel for schedule(static) if (K > 1)
    for (int64_t offset = 0; offset < K; ++offset) {
        for (int64_t i = offset + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
