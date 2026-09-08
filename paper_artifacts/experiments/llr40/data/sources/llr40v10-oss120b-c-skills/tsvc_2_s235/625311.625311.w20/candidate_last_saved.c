#include <stdint.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    // Update a[i] for all i
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
    }
    // Compute aa using recurrence over rows, vectorizable over columns (i)
    for (int64_t j = 1; j < LEN_2D; ++j) {
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * a[i];
        }
    }
}
