#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 1) return;

    if (n < 64) {
        for (int64_t j = 1; j < n; ++j) {
            const double *restrict prev = &aa[(j - 1) * n];
            const double *restrict b = &bb[j * n];
            double *restrict cur = &aa[j * n];
            #pragma omp simd
            for (int64_t i = 0; i < n; ++i) {
                cur[i] = prev[i] + b[i];
            }
        }
        return;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const int64_t i0 = (n * (int64_t)tid) / nth;
        const int64_t i1 = (n * (int64_t)(tid + 1)) / nth;
        const int64_t len = i1 - i0;

        double *restrict cur = &aa[n + i0];
        const double *restrict prev = &aa[i0];
        const double *restrict b = &bb[n + i0];

        for (int64_t j = 1; j < n; ++j) {
            #pragma omp simd
            for (int64_t i = 0; i < len; ++i) {
                cur[i] = prev[i] + b[i];
            }
            prev = cur;
            cur += n;
            b += n;
        }
    }
}
