#include <stdint.h>
#include <stdio.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    fprintf(stderr, "LEN_2D = %ld\n", (long)LEN_2D);
    const int64_t n = LEN_2D;
    for (int64_t i = 8; i < n; ++i) {
        for (int64_t j = 8; j < n; ++j) {
            aa[j * n + i] = aa[(j - 1) * n + i] + cc[j * n + i];
        }
        for (int64_t j = 8; j < n; ++j) {
            bb[j * n + i] = bb[j * n + (i - 1)] + cc[j * n + i];
        }
    }
}
