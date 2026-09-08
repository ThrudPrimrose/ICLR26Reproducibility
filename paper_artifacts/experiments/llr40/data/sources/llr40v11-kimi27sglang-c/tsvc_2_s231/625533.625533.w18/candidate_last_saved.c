#include <stdint.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t BLOCK = 1024;

    #pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < LEN_2D; ii += BLOCK) {
        const int64_t i_end = (ii + BLOCK < LEN_2D) ? ii + BLOCK : LEN_2D;
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const double *prev = &aa[(j - 1) * LEN_2D];
            const double *b = &bb[j * LEN_2D];
            double *cur = &aa[j * LEN_2D];
            #pragma omp simd
            for (int64_t i = ii; i < i_end; ++i) {
                cur[i] = prev[i] + b[i];
            }
        }
    }
}
