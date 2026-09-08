#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <float.h>
#include <limits.h>

/* Optimized version of tsvc_2_s318 kernel. Finds the maximum absolute value in array a with stride inc, and returns max+index of first occurrence as double in result[0]. Uses OpenMP parallel reduction on a custom struct to keep both the max value and the corresponding index. */


void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    double maxv = -INFINITY;
    int64_t index = INT64_MAX;
    #pragma omp parallel
    {
        double local_max = -INFINITY;
        int64_t local_idx = INT64_MAX;
        #pragma omp for nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = fabs(a[i * inc]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            } else if (v == local_max && i < local_idx) {
                local_idx = i;
            }
        }
        #pragma omp critical
        {
            if (local_max > maxv) {
                maxv = local_max;
                index = local_idx;
            } else if (local_max == maxv && local_idx < index) {
                index = local_idx;
            }
        }
    }
    result[0] = maxv + (double)index;
}

