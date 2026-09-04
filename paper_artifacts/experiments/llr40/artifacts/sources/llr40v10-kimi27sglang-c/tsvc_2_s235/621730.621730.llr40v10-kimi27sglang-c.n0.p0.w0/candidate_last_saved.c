#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;

#pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
        a[i] += b[i] * c[i];
    }

    for (int64_t j = 1; j < n; ++j) {
        const int64_t row = j * n;
        const int64_t prev = row - n;
#pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            aa[row + i] = aa[prev + i] + bb[row + i] * a[i];
        }
    }
}
