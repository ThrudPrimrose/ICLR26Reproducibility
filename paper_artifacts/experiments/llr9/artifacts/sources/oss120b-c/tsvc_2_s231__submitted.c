#include <stddef.h>
#include <stdint.h>

void tsvc_2_s231_fp64(double *aa, double *bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    // Ignore workspace arguments (not used)
    (void)workspace;
    (void)workspace_bytes;
    // Loop interchange: outer loop over rows (j) for dependency,
    // inner loop over columns (i) to enable vectorization.
    for (int64_t j = 1; j < LEN_2D; ++j) {
        double *a_row = aa + j * LEN_2D;
        const double *b_row = bb + j * LEN_2D;
        const double *a_prev = aa + (j - 1) * LEN_2D;
        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a_row[i] = a_prev[i] + b_row[i];
        }
    }
}
