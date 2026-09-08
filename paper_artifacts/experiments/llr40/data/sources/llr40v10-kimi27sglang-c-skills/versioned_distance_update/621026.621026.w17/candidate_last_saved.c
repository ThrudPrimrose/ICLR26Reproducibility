#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a, double *restrict b, double *restrict c,
                                    int64_t LEN_1D, int64_t K,
                                    uint8_t *restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    if (K <= 0 || LEN_1D <= K) return;

    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < K; p++) {
        for (int64_t i = p + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
