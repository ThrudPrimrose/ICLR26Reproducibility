#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a, const double *restrict b,
                                    const double *restrict c, const int64_t LEN_1D,
                                    const int64_t K) {
    for (int64_t i = K; i < LEN_1D; ++i) {
        a[i] = 0.75 * a[i - K] + b[i] * c[i];
    }
}
