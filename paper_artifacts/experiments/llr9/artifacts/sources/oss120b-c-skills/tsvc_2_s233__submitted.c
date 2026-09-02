#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    // Update aa: column-wise recurrence over rows (parallel over columns)
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        for (int64_t j = 8; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    // Update bb: row-wise recurrence over columns (parallel over rows)
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < LEN_2D; ++j) {
        for (int64_t i = 8; i < LEN_2D; ++i) {
            bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
        }
    }
}
