#include <stdint.h>
#include <cblas.h>
void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    for (int64_t j = 0; j < LEN_2D; j++) {
        int64_t n = LEN_2D - j - 1;
        if (n > 0) {
            cblas_daxpy((int)n, -a[j], &aa[j * LEN_2D + j + 1], 1, &a[j + 1], 1);
        }
    }
}
