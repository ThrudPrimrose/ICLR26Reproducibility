#include <stdint.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    // Vectorize the inner loop. The outer loop is inherently sequential due to dependencies.
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double a_j = a[j];                    // load scalar once
        const double *restrict row = aa + j * LEN_2D + (j + 1);
        double *restrict a_ptr = a + (j + 1);
        int64_t n = LEN_2D - (j + 1);
        #pragma omp simd
        for (int64_t k = 0; k < n; ++k) {
            a_ptr[k] -= row[k] * a_j;
        }
    }
}
