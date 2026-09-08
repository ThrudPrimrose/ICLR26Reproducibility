/* Optimized version of ext_break_capture kernel.
 * Find the first index i where a[i] > k (k = 1.0) and capture the value.
 * Parallel reduction to compute the minimum index that satisfies the condition.
 */
#include <stdint.h>
#include <omp.h>

void ext_break_capture_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0;
    // Sentinel value representing "not found"
    const int64_t sentinel = INT64_MAX;
    int64_t best = sentinel;

    #pragma omp parallel for reduction(min:best) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > k) {
            best = i; // reduction will keep the smallest i across all threads
        }
    }

    if (best != sentinel) {
        out_index[0] = best;
        out_value[0] = a[best];
    } else {
        out_index[0] = -1;
        out_value[0] = -1.0;
    }
}
