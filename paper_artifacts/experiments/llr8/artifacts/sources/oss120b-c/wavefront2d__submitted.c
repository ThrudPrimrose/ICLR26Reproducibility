#include <omp.h>
#include <stdint.h>
#include <stddef.h>

// 2D wavefront relaxation kernel (in-place averaging) using anti-diagonal parallelism.

void wavefront2d_fp64(double *a, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    const int64_t max_sum = 2 * (LEN_2D - 1);
    #pragma omp parallel
    {
        for (int64_t sum = 2; sum <= max_sum; ++sum) {
            int64_t i_start = sum - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = sum - 1;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = sum - i;
                int64_t idx = i * LEN_2D + j;
                int64_t idx_up = (i - 1) * LEN_2D + j;
                int64_t idx_left = i * LEN_2D + (j - 1);
                int64_t idx_ul = (i - 1) * LEN_2D + (j - 1);
                a[idx] = 0.25 * (a[idx] + a[idx_up] + a[idx_left] + a[idx_ul]);
            }
        }
    }
}
