#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double a_j = a[j];
        const double *aa_row = aa + j * LEN_2D + (j + 1);
        double *a_ptr = a + (j + 1);
        int64_t n = LEN_2D - (j + 1);
        #pragma omp simd
        for (int64_t k = 0; k < n; ++k) {
            a_ptr[k] -= aa_row[k] * a_j;
        }
    }
}
