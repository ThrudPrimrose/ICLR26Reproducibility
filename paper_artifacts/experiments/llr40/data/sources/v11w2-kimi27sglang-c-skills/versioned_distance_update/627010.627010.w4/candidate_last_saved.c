#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double * restrict a, double * restrict b, double * restrict c, int64_t K, int64_t LEN_1D, uint8_t * restrict workspace, int64_t workspace_bytes)
{
    const double decay = 0.75;
    if (K <= 0 || K >= LEN_1D) return;
    for (int64_t i = K; i < LEN_1D; ++i) {
        a[i] = decay * a[i - K] + b[i] * c[i];
    }
}
