#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 0) return;
    if (n < 16384) {
        a[0] = b[0] * c[0];
        #pragma omp simd
        for (int64_t i = 1; i < n; ++i) {
            a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
        }
        return;
    }
    a[0] = b[0] * c[0];
    const int64_t q = n / 8;
    const int64_t base = q - 1;
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < base; ++i) {
        a[1 + i] = b[1 + i] * c[1 + i] + b[i] * c[i];
        a[1 + 1 * q + i] = b[1 + 1 * q + i] * c[1 + 1 * q + i] + b[1 * q + i] * c[1 * q + i];
        a[1 + 2 * q + i] = b[1 + 2 * q + i] * c[1 + 2 * q + i] + b[2 * q + i] * c[2 * q + i];
        a[1 + 3 * q + i] = b[1 + 3 * q + i] * c[1 + 3 * q + i] + b[3 * q + i] * c[3 * q + i];
        a[1 + 4 * q + i] = b[1 + 4 * q + i] * c[1 + 4 * q + i] + b[4 * q + i] * c[4 * q + i];
        a[1 + 5 * q + i] = b[1 + 5 * q + i] * c[1 + 5 * q + i] + b[5 * q + i] * c[5 * q + i];
        a[1 + 6 * q + i] = b[1 + 6 * q + i] * c[1 + 6 * q + i] + b[6 * q + i] * c[6 * q + i];
        a[1 + 7 * q + i] = b[1 + 7 * q + i] * c[1 + 7 * q + i] + b[7 * q + i] * c[7 * q + i];
    }
    for (int64_t p = 1; p <= 7; ++p) {
        const int64_t j = p * q;
        if (j < n) a[j] = b[j] * c[j] + b[j - 1] * c[j - 1];
    }
    for (int64_t j = 8 * q; j < n; ++j) {
        a[j] = b[j] * c[j] + b[j - 1] * c[j - 1];
    }
}
