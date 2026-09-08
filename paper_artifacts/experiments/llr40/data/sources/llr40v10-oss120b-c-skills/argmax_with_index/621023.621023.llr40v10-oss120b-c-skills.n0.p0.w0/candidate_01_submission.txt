/*
 * Optimized version of argmax_with_index kernel.
 * Finds the maximum value in an array and its index.
 * Parallel reduction using OpenMP to speed up for large arrays.
 */

#include <stdint.h>
#include <math.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    // Edge case: empty array (should not happen). Return zero.
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = -1;
        return;
    }

    // Initialize with first element.
    double x = a[0];
    int64_t idx = 0;

    // Parallel reduction: each thread finds its local max and index.
    #pragma omp parallel
    {
        double local_max = -INFINITY;
        int64_t local_idx = -1;

        #pragma omp for nowait
        for (int64_t i = 1; i < LEN_1D; ++i) {
            double v = a[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        // Combine thread-local results.
        #pragma omp critical
        {
            if (local_max > x || (local_max == x && local_idx < idx)) {
                x = local_max;
                idx = local_idx;
            }
        }
    }

    out_value[0] = x;
    out_index[0] = idx;
}
