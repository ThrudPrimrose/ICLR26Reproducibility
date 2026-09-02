#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
        for (int64_t j = 1; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * a[i];
        }
    }
}
