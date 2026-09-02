#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D,
                      uint8_t *restrict workspace, int64_t workspace_bytes) {
    const int64_t n = LEN_1D;
    (void)workspace;
    (void)workspace_bytes;
    if (n <= 0) return;
    const double c = 0.333;

    if (n == 1) {
        a[0] = (b[0] + b[0] + b[0]) * c;
        return;
    }
    if (n == 2) {
        a[0] = (b[0] + b[1] + b[0]) * c;
        a[1] = (b[1] + b[0] + b[1]) * c;
        return;
    }

    a[0] = (b[0] + b[n - 1] + b[n - 2]) * c;
    a[1] = (b[1] + b[0] + b[n - 1]) * c;

    #pragma omp parallel for simd safelen(64) schedule(static)
    for (int64_t i = 2; i < n; i += 1) {
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * c;
    }
}
