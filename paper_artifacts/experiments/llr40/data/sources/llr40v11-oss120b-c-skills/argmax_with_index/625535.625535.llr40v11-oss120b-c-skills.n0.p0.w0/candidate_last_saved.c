/* argmax_with_index_fp64: find max value and index in a double array.
   Optimized parallel version using OpenMP.
   Signature matches the reference exactly. */

#include <stdint.h>
#include <math.h>
#include <omp.h>

/* Find the maximum value and its first occurrence index. */
void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    /* Guard against empty input (should not occur per spec). */
    if (LEN_1D <= 0) {
        out_value[0] = -INFINITY;
        out_index[0] = -1;
        return;
    }

    double global_max = -INFINITY;
    int64_t global_idx = INT64_MAX;

    #pragma omp parallel
    {
        double local_max = -INFINITY;
        int64_t local_idx = INT64_MAX;
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = a[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            } else if (v == local_max && i < local_idx) {
                local_idx = i; // keep smallest index for ties
            }
        }
        /* Combine thread-local results into the global result. */
        #pragma omp critical
        {
            if (local_max > global_max) {
                global_max = local_max;
                global_idx = local_idx;
            } else if (local_max == global_max && local_idx < global_idx) {
                global_idx = local_idx;
            }
        }
    }
    out_value[0] = global_max;
    out_index[0] = global_idx;
}
