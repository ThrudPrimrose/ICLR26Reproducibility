#include <stdint.h>
#include <omp.h>

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 1) return;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int64_t j = 1; j < LEN_2D; ++j) {
        const int64_t base = j * LEN_2D;
        double prev = aa[base];
        for (int64_t i = 1; i <= j; ++i) {
            double x = prev * prev + bb[base + i];
            aa[base + i] = x;
            prev = x;
        }
    }
}
