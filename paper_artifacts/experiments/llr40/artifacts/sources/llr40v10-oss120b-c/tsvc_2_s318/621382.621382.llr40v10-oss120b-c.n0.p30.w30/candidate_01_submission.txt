/* Optimized version of tsvc_2_s318 kernel.
 * Computes the sum of the maximum absolute value in array a and its index.
 * Uses OpenMP parallel reduction to find the maximum, then a sequential scan to locate the first occurrence.
 */

#include <math.h>
#include <stdint.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                       const int64_t LEN_1D, const int64_t inc) {
    // Guard against empty input (should not happen in benchmark).
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }

    double maxv = 0.0;
    // First element initializes maxv to fabs(a[0]). This matches reference behavior.
    maxv = fabs(a[0]);

    // Parallel reduction to find maximum absolute value.
    // OpenMP reduction with max works for floating-point types.
    #pragma omp parallel for reduction(max:maxv) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v > maxv) {
            maxv = v;
        }
    }

    // Find the first index where the maximum occurs using a parallel reduction.
    int64_t index = LEN_1D; // sentinel large value
    #pragma omp parallel for reduction(min:index) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (fabs(a[i * inc]) == maxv) {
            if (i < index) index = i;
        }
    }
    // In case maxv wasn't found (should not happen), fallback to 0.
    if (index == LEN_1D) index = 0;

    result[0] = maxv + (double)index;
}
