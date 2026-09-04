#include <stdint.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Compute aa prefix sum down rows for each column (parallel over columns)
    #pragma omp parallel
    {
        // Compute aa prefix sum down rows for each column (parallel over columns)
        #pragma omp for schedule(static)
        for (int64_t i = 8; i < LEN_2D; ++i) {
            for (int64_t j = 8; j < LEN_2D; ++j) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
            }
        }

        // Compute bb prefix sum across columns for each row (parallel over rows)
        #pragma omp for schedule(static)
        for (int64_t j = 8; j < LEN_2D; ++j) {
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
            }
        }
    }
}
