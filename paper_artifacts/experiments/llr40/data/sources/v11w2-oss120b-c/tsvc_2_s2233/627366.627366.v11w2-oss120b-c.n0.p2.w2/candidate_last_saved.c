#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // First loop: compute aa column-wise prefix sum across rows (j) for each column i.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        // Process rows j from 8 to LEN_2D-1.
        for (int64_t j = 8; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    // Second loop: compute bb row-wise prefix sum across columns (i) for each row j.
    // Here we swap loop order to enable parallelism over independent rows.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < LEN_2D; ++j) {
        for (int64_t i = 8; i < LEN_2D; ++i) {
            bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}
