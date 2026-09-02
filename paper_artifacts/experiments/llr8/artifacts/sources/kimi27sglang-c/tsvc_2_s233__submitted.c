#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, double *restrict cc,
                      int64_t LEN_2D, uint8_t *restrict workspace, int64_t workspace_bytes)
{
    const int64_t n = LEN_2D;

    /* aa: recurrence down rows.  Row-major -> make the recurrence the
       outer loop and the independent axis (columns) the vectorised inner loop. */
    for (int64_t j = 8; j < n; ++j) {
        const double *restrict crow = cc + j * n;
        const double *restrict aprev = aa + (j - 1) * n;
        double *restrict arow = aa + j * n;
        #pragma omp simd
        for (int64_t i = 8; i < n; ++i) {
            arow[i] = aprev[i] + crow[i];
        }
    }

    /* bb: recurrence along columns.  Each row is an independent prefix sum,
       so parallelise rows and run the recurrence serially inside a row. */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < n; ++j) {
        const double *restrict crow = cc + j * n;
        double *restrict brow = bb + j * n;
        double prev = brow[7];
        for (int64_t i = 8; i < n; ++i) {
            prev = prev + crow[i];
            brow[i] = prev;
        }
    }
}
