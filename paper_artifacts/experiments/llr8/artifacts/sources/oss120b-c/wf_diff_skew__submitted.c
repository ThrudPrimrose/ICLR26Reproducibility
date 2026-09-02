#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *a, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    // Assumes a is a contiguous LEN_2D x LEN_2D array in row-major order.
    // Compute: a[i][j] = a[i][j] + a[i-1][j] + a[i-1][j+1]
    // i from 1 to LEN_2D-1, j from 0 to LEN_2D-2.
    #pragma omp parallel
    for (int i = 1; i < LEN_2D; ++i) {
        double *row_i = a + i * (size_t)LEN_2D;
        double *row_im1 = a + (i - 1) * (size_t)LEN_2D;
        // Vectorize inner loop.
        #pragma omp for schedule(static)
        for (int j = 0; j < LEN_2D - 1; ++j) {
            row_i[j] = row_i[j] + row_im1[j] + row_im1[j + 1];
        }
    }
}
