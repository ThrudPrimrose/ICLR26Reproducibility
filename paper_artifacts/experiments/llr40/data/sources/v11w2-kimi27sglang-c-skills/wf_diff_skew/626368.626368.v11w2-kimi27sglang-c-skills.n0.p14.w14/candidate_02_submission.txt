#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    #pragma omp parallel
    {
        for (int64_t i = 1; i < n; ++i) {
            double *restrict row_curr = a + i * n;
            const double *restrict row_prev = a + (i - 1) * n;
            #pragma omp for simd schedule(static)
            for (int64_t j = 0; j < n - 1; ++j) {
                row_curr[j] = row_curr[j] + row_prev[j] + row_prev[j + 1];
            }
        }
    }
}
