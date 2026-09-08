/* Optimized version of tsvc_2_s1244_fp64.
   Copies original 'a' to a temporary buffer to avoid anti-dependence,
   then computes 'a' and 'd' in separate loops. Uses OpenMP to parallelize.
*/

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                       double *restrict d, const int64_t LEN_1D) {
    // Allocate temporary buffer to hold original a values.
    // Use malloc; the overhead is small compared to compute for large LEN.
    double *orig = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!orig) {
        // Allocation failed; fallback to serial version without parallelism.
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            d[i] = a[i] + a[i + 1];
        }
        return;
    }

    // Parallel region – reuse threads for all three loops.
    #pragma omp parallel
    {
        // Copy original a into temporary buffer.
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            orig[i] = a[i];
        }
        // Compute new a values (skip the last element to match reference).
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
        }
        // Compute d using the stored original a values.
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            d[i] = a[i] + orig[i + 1];
        }
    }
    free(orig);
}

