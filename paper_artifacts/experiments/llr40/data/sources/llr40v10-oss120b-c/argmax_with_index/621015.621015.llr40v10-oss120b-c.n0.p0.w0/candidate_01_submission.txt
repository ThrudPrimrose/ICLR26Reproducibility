/* Optimized implementation of argmax_with_index_fp64 kernel.
 * Computes the maximum value in an array and the index of its first occurrence.
 * Uses OpenMP parallel reduction for max value and parallel min reduction for index.
 */

#include <stdint.h>
#include <omp.h>
#include <float.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        // Undefined behavior for empty array; set defaults.
        out_value[0] = 0.0;
        out_index[0] = -1;
        return;
    }

    // Compute maximum value using OpenMP reduction (max).
    // Initialize with a[0] to ensure the element is considered.
    double max_val = a[0];
    #pragma omp parallel for reduction(max:max_val) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > max_val) {
            max_val = v;
        }
    }

    // Find the first index of max_val using a min reduction of indices.
    // Use sentinel LEN_1D (out of range) as initial value.
    int64_t min_idx = LEN_1D;
    #pragma omp parallel for reduction(min:min_idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] == max_val && i < min_idx) {
            min_idx = i;
        }
    }

    // If for some reason the value wasn't found (shouldn't happen), fallback to 0.
    if (min_idx == LEN_1D) {
        min_idx = 0;
    }

    out_value[0] = max_val;
    out_index[0] = min_idx;
}

