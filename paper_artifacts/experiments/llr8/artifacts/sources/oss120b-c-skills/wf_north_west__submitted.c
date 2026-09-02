#include <stdint.h>
#include <omp.h>

void wf_north_west_fp64(double * restrict a, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    #pragma omp parallel
    {
        for (int64_t k = 2; k <= 2 * (LEN_2D - 1); ++k) {
            int64_t i_start = k - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = k - 1;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = k - i;
                a[i * LEN_2D + j] = a[i * LEN_2D + j] + a[(i - 1) * LEN_2D + j] + a[i * LEN_2D + (j - 1)];
            }
        }
    }
}
