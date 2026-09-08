#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n < 2) return;
    const int64_t m = n - 1;               /* number of columns updated per row */
    const int64_t B = 64;                  /* tile rows */
    const int64_t W = 512;                 /* tile cols; W >= B */

    #pragma omp parallel
    {
        for (int64_t i0 = 1; i0 < n; i0 += B) {
            const int64_t i1 = (i0 + B < n) ? (i0 + B) : n;
            #pragma omp for schedule(static)
            for (int64_t j0 = 0; j0 < m; j0 += W) {
                const int64_t j1 = (j0 + W < m) ? (j0 + W) : m;
                for (int64_t i = i0; i < i1; ++i) {
                    double *restrict cur = a + i * n;
                    const double *restrict prev = a + (i - 1) * n;
                    #pragma omp simd
                    for (int64_t j = j0; j < j1; ++j) {
                        cur[j] = cur[j] + prev[j] + prev[j + 1];
                    }
                }
            }
        }
    }
}
