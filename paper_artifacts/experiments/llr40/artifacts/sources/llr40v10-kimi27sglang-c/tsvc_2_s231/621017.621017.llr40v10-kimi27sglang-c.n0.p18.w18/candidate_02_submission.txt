#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
            }
        }
    }
}
