#include <stdint.h>

void versioned_distance_update_fp64(double *restrict a, double *restrict b, double *restrict c,
                                    const int64_t K, const int64_t LEN_1D,
                                    uint8_t *workspace, const int64_t workspace_size) {
    (void)workspace; (void)workspace_size;
    for (int64_t i = K; i < LEN_1D; ++i)
        a[i] = 0.75 * a[i - K] + b[i] * c[i];
}
