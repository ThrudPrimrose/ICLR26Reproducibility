#include <stdint.h>
#include <stddef.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    // Simple sequential prefix sum with manual unrolling for performance.
    if (LEN_1D <= 0) return;
    double sum = 0.0;
    int64_t i = 0;
    int64_t n8 = LEN_1D & ~7LL; // largest multiple of 8 <= LEN_1D
    for (; i < n8; i += 8) {
        sum += a[i];
        b[i] = sum;
        sum += a[i + 1];
        b[i + 1] = sum;
        sum += a[i + 2];
        b[i + 2] = sum;
        sum += a[i + 3];
        b[i + 3] = sum;
        sum += a[i + 4];
        b[i + 4] = sum;
        sum += a[i + 5];
        b[i + 5] = sum;
        sum += a[i + 6];
        b[i + 6] = sum;
        sum += a[i + 7];
        b[i + 7] = sum;
    }
    for (; i < LEN_1D; ++i) {
        sum += a[i];
        b[i] = sum;
    }
}
