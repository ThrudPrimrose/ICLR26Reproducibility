#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    const double s = 0.333;

    if (n == 0) return;
    if (n == 1) {
        const double v = b[0];
        a[0] = (v + v + v) * s;
        return;
    }
    if (n == 2) {
        const double b0 = b[0];
        const double b1 = b[1];
        a[0] = (b0 + b1 + b0) * s;
        a[1] = (b1 + b0 + b1) * s;
        return;
    }

    a[0] = (b[0] + b[n - 1] + b[n - 2]) * s;
    a[1] = (b[1] + b[0] + b[n - 1]) * s;

    if (n < 4096) {
        for (int64_t i = 2; i < n; i++) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * s;
        }
    } else {
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 2; i < n; i++) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * s;
        }
    }
}
