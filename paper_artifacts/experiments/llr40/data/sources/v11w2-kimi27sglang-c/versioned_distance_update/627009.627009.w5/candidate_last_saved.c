#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a, double *restrict b,
                                    double *restrict c, int64_t K, int64_t LEN_1D,
                                    uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    if (K == 1) {
        for (int64_t i = 1; i < LEN_1D; i++) {
            a[i] = 0.75 * a[i - 1] + b[i] * c[i];
        }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < K; j++) {
        for (int64_t i = j + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
