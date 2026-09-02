#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    // Workspace not used in this kernel.
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double aj = a[j];
        const double *aa_row = aa + j * LEN_2D;
        #pragma omp simd
        for (int64_t i = j + 1; i < LEN_2D; ++i) {
            a[i] -= aa_row[i] * aj;
        }
    }
}
