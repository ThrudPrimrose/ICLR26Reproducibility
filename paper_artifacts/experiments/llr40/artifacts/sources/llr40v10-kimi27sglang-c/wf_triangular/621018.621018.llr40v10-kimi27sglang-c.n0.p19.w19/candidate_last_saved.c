#include <stdint.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t dmax = 2 * (LEN_2D - 1);
    #pragma omp parallel
    {
        for (int64_t d = 2; d <= dmax; ++d) {
            int64_t i_start = d - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = d / 2;
            if (i_end >= LEN_2D) i_end = LEN_2D - 1;

            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = d - i;
                a[i * LEN_2D + j] = a[i * LEN_2D + j] + a[(i - 1) * LEN_2D + j] + a[i * LEN_2D + (j - 1)];
            }
        }
    }
}
