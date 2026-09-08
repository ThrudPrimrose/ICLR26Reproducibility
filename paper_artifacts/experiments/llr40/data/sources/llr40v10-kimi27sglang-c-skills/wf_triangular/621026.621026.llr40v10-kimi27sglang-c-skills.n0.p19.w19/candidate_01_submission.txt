#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return;

    #pragma omp parallel
    {
        for (int64_t s = 2; s < 2 * N; ++s) {
            int64_t i_min = 1;
            int64_t i_max = N - 1;
            if (s - (N - 1) > i_min) i_min = s - (N - 1);
            if (s / 2 < i_max) i_max = s / 2;

            #pragma omp for schedule(static)
            for (int64_t i = i_min; i <= i_max; ++i) {
                int64_t j = s - i;
                a[i * N + j] = a[i * N + j] + a[(i - 1) * N + j] + a[i * N + (j - 1)];
            }
        }
    }
}
