/* Optimized version of tsvc_2_s318_fp64 using OpenMP parallelism.
   Computes the maximum absolute value and its first index in a strided array.
   Returns result[0] = maxv + (double)index.
*/

#include <math.h>
#include <stdint.h>
#include <omp.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    if (LEN_1D <= 0) {
        result[0] = 0.0;
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
            double v = fabs(a[i * inc]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            } else if (v == local_max && i < local_idx) {
                local_idx = i; // tie – keep smaller index
            }
        }
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
    result[0] = global_max + (double)global_idx;
}
