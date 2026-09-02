/*
 * TSVC_2 kernel s275 implementation.
 *
 * Reference NumPy implementation (see /shared/tasks/tsvc_2_s275/tsvc_2_s275_numpy.py):
 *   for i in range(LEN_2D):
 *       if aa[0, i] > 0.0:
 *           for j in range(1, LEN_2D):
 *               aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]
 *
 * Matrices are stored in row-major order: element (row, col) is at
 *   ptr[row * LEN_2D + col]
 *
 * Optimisation strategy:
 *   1. Pre-compute a column mask for the condition aa[0,i] > 0.
 *   2. Iterate rows outermost and columns innermost for unit stride.
 *   3. Use OpenMP SIMD to encourage vectorisation.
 *   4. Declare pointers as restrict.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s275_fp64(double * restrict aa,
          const double * restrict bb,
          const double * restrict cc,
          const int64_t LEN_2D,
          uint8_t * restrict workspace,
          const int64_t workspace_bytes)
{
    if (LEN_2D <= 0) {
        return;
    }

    char *mask = (char *)malloc((size_t)LEN_2D);

    #pragma omp simd
    for (int64_t i = 0; i < LEN_2D; ++i) {
        mask[i] = (aa[i] > 0.0) ? 1 : 0;
    }

    for (int64_t j = 1; j < LEN_2D; ++j) {
        const double *bb_row = bb + j * LEN_2D;
        const double *cc_row = cc + j * LEN_2D;
        double *aa_row = aa + j * LEN_2D;
        const double *aa_prev = aa + (j - 1) * LEN_2D;

        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (mask[i]) {
                aa_row[i] = aa_prev[i] + bb_row[i] * cc_row[i];
            }
        }
    }
    free(mask);

}
