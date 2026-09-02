#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;
    // Handle first element separately (t is initially zero)
    a[0] = b[0] * c[0];
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double s = b[i] * c[i];
        double prev = b[i-1] * c[i-1];
        a[i] = s + prev;
    }
}
