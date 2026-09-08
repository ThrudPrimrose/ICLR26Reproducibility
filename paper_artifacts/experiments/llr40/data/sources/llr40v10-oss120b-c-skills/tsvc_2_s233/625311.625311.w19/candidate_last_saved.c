/* Optimized version of tsvc_2_s233 kernel.
   Original reference: /shared/tasks/tsvc_2_s233/tsvc_2_s233_reference.c */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Compute aa: recurrence across rows (j) for each column i.
    // Columns are independent, so we parallelise the outer i loop.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        #pragma omp simd
        for (int64_t j = 8; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    // Compute bb: prefix sum across columns (i) for each row j.
    // Rows are independent, so we parallelise the outer j loop.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < LEN_2D; ++j) {
        for (int64_t i = 8; i < LEN_2D; ++i) {
            bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
        }
    }
}
