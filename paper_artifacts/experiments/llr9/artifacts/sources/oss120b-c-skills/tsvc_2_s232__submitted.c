#include <stdint.h>
#include <omp.h>

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    #pragma omp parallel for schedule(static)
    for (int64_t j = 1; j < LEN_2D; ++j) {
        for (int64_t i = 1; i <= j; ++i) {
            double prev = aa[j * LEN_2D + (i - 1)];
            aa[j * LEN_2D + i] = prev * prev + bb[j * LEN_2D + i];
        }
    }
}
