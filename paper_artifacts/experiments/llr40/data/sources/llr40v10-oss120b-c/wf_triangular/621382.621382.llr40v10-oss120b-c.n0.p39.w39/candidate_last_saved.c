#include <stdint.h>

#include <stdint.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    // Wavefront parallelization over anti-diagonals (i + j constant)
    const int64_t max_sum = 2 * (LEN_2D - 1);
    #pragma omp parallel
    {
        for (int64_t sum = 2; sum <= max_sum; ++sum) {
            int64_t i_min = sum - (LEN_2D - 1);
            if (i_min < 1) i_min = 1;
            int64_t i_max = sum / 2;
            if (i_max > (LEN_2D - 1)) i_max = LEN_2D - 1;
            if (i_min > i_max) continue;
            #pragma omp for schedule(static)
            for (int64_t i = i_min; i <= i_max; ++i) {
                int64_t j = sum - i;
                a[i * LEN_2D + j] = a[i * LEN_2D + j] + a[(i - 1) * LEN_2D + j] + a[i * LEN_2D + (j - 1)];
            }
            // implicit barrier at end of omp for
        }
    }
}
