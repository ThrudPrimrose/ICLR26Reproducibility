#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    // Allocate temporary buffer to hold original a values
    double *a_orig = (double*)aligned_alloc(64, (size_t)LEN_1D * sizeof(double));
    if (a_orig == NULL) {
        // Allocation failed; fallback to serial computation
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            d[i] = a[i] + a[i+1];
        }
        return;
    }
    // Copy original a and compute a and d in parallel
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a_orig[i] = a[i];
        }
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            double a_val = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            a[i] = a_val;
            d[i] = a_val + a_orig[i + 1];
        }
    }
    free(a_orig);
}
