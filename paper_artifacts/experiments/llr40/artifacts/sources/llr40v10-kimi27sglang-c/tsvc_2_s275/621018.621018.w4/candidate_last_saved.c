#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 1) return;

    for (int64_t j = 1; j < n; j++) {
        const double *restrict prev = aa + (j - 1) * n;
        double *restrict cur = aa + j * n;
        const double *restrict brow = bb + j * n;
        const double *restrict crow = cc + j * n;

        #pragma omp simd
        for (int64_t i = 0; i < n; i++) {
            const double tmp = prev[i] + brow[i] * crow[i];
            if (aa[i] > 0.0) cur[i] = tmp;
        }
    }
}
