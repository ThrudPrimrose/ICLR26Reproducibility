#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s233_fp64(double * restrict aa, double * restrict bb, double * restrict cc, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t n = LEN_2D;
    if (n <= 8) return;
    const int64_t B = 512;

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t cols = n - 8;
        const int64_t i0 = 8 + (tid * cols) / nt;
        const int64_t i1 = 8 + ((tid + 1) * cols) / nt;
        const int64_t width = i1 - i0;
        double * restrict a_prev = (double *)malloc(width * sizeof(double));

        for (int64_t j0 = 8; j0 < n; j0 += B) {
            int64_t j1 = j0 + B;
            if (j1 > n) j1 = n;

            // initialise previous-row values for this column block
            const double * restrict north = aa + (j0 - 1) * n + i0;
            for (int64_t k = 0; k < width; k++) {
                a_prev[k] = north[k];
            }

            for (int64_t j = j0; j < j1; j++) {
                const double * restrict cc_row = cc + j * n + i0;
                double * restrict aa_row = aa + j * n + i0;
                #pragma omp simd
                for (int64_t k = 0; k < width; k++) {
                    double v = a_prev[k] + cc_row[k];
                    aa_row[k] = v;
                    a_prev[k] = v;
                }
            }

            #pragma omp barrier
        }

        free(a_prev);
    }

    // bb: recurrence across columns; rows are independent.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < n; j++) {
        const double * restrict cc_row = cc + j * n;
        double * restrict bb_row = bb + j * n;
        double v = bb_row[7];
        for (int64_t i = 8; i < n; i++) {
            v = v + cc_row[i];
            bb_row[i] = v;
        }
    }
}
