#include <stdint.h>
#include <omp.h>

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    #pragma omp parallel for schedule(nonmonotonic:guided)
    for (int64_t j = 1; j < LEN_2D; ++j) {
        double *restrict arow = aa + j * LEN_2D;
        const double *restrict brow = bb + j * LEN_2D;
        double prev = arow[0];
        for (int64_t i = 1; i <= j; ++i) {
            prev = prev * prev + brow[i];
            arow[i] = prev;
        }
    }
}
