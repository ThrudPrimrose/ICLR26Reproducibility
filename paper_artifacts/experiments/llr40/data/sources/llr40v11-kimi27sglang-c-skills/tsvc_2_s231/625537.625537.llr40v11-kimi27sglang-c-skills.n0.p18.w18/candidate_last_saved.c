#include <stdint.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 64
#endif

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    #pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < n; ii += BLOCK) {
        const int64_t len = (ii + BLOCK <= n) ? BLOCK : (n - ii);
        double tmp[BLOCK];
        #pragma omp simd
        for (int64_t i = 0; i < len; ++i) {
            tmp[i] = aa[ii + i];
        }
        for (int64_t j = 1; j < n; ++j) {
            const double *restrict brow = bb + j * n + ii;
            double *restrict arow = aa + j * n + ii;
            #pragma omp simd
            for (int64_t i = 0; i < len; ++i) {
                tmp[i] += brow[i];
                arow[i] = tmp[i];
            }
        }
    }
}
