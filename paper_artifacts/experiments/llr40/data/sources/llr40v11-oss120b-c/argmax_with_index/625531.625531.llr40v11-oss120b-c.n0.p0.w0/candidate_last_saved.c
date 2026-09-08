/* Optimized version of argmax_with_index for double precision.
 * Computes the maximum value in an array and its index.
 * Utilizes OpenMP for parallel reduction across threads.
 */

#include <stddef.h>
#include <stdint.h>
#include <float.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    // Edge case: empty array not defined; assume LEN_1D >= 1.
    if (LEN_1D <= 0) {
        out_value[0] = -INFINITY;
        out_index[0] = -1;
        return;
    }

    double global_max = a[0];
    int64_t global_idx = 0;

    #pragma omp parallel
    {
        double local_max = a[0];
        int64_t local_idx = 0;
        #pragma omp for nowait schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = a[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        #pragma omp critical
        {
            if (local_max > global_max) {
                global_max = local_max;
                global_idx = local_idx;
            }
        }
    }

    out_value[0] = global_max;
    out_index[0] = global_idx;
}
