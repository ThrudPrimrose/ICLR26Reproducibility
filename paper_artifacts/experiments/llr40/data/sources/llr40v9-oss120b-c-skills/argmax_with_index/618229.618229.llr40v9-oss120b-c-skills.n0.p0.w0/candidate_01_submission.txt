#include <stdint.h>
#include <omp.h>
#include <math.h>

/*
 * argmax_with_index_fp64: find the maximum value in array a of length LEN_1D,
 * and store the max value in out_value[0] and its index in out_index[0].
 * Uses OpenMP parallelization to speed up on multi-core systems.
 */
void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    // Guard against empty array (should not happen per spec)
    if (LEN_1D <= 0) {
        out_value[0] = -INFINITY;
        out_index[0] = -1;
        return;
    }

    // Result holder
    double max_val = a[0];
    int64_t max_idx = 0;

    // Parallel region: each thread finds its local maximum and index.
    #pragma omp parallel
    {
        double local_max = -INFINITY;
        int64_t local_idx = -1;
        // Distribute the loop iterations among threads.
        #pragma omp for nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double ai = a[i];
            if (ai > local_max) {
                local_max = ai;
                local_idx = i;
            }
        }
        // Combine local results into the shared result.
        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
                max_idx = local_idx;
            }
        }
    }

    out_value[0] = max_val;
    out_index[0] = max_idx;
}

