#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    #pragma omp parallel
    {
        for (int64_t i = 1; i < N; ++i) {
            double *restrict cur = a + i * N;
            double *restrict prev = a + (i - 1) * N;
            #pragma omp for simd schedule(static)
            for (int64_t j = 0; j < N - 1; ++j) {
                cur[j] += prev[j] + prev[j + 1];
            }
        }
    }
}
