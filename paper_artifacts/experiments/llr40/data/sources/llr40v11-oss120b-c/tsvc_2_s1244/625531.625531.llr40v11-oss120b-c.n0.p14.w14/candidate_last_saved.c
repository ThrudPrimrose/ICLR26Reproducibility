#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    double *a_old = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!a_old) {
        // Fallback sequential version.
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            double bi = b[i];
            double ci = c[i];
            a[i] = bi + ci * ci + bi * bi + ci;
            d[i] = a[i] + a[i + 1];
        }
        return;
    }

    #pragma omp parallel
    {
        // Copy original a values.
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a_old[i] = a[i];
        }

        // Compute new a values (for i < LEN_1D-1).
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            double bi = b[i];
            double ci = c[i];
            a[i] = bi + ci * ci + bi * bi + ci;
        }

        // Compute d using new a and old a.
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            d[i] = a[i] + a_old[i + 1];
        }
    }

    free(a_old);
}
