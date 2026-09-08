#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        for (int64_t s = 2; s <= 2 * (LEN_2D - 1); ++s) {
            int64_t i_start = s - LEN_2D + 1;
            if (i_start < 1) i_start = 1;
            int64_t i_end = s / 2;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = s - i;
                a[i * LEN_2D + j] += a[(i - 1) * LEN_2D + j] + a[i * LEN_2D + (j - 1)];
            }
        }
    }
}
