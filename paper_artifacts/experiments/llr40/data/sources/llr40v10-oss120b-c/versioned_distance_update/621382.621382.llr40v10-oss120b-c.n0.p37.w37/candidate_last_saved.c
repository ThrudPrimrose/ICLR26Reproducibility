/* Handwritten C implementation of versioned_distance_update kernel. */
#include <stdint.h>

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t LEN_1D,
                                    const int64_t K) {
    /* Support both (LEN, K) and (K, LEN) argument orders. */
    int64_t len = LEN_1D;
    int64_t k = K;
    if (len < k) {
        int64_t tmp = len;
        len = k;
        k = tmp;
    }
    if (k <= 1) {
        for (int64_t i = k; i < len; ++i) {
            a[i] = 0.75 * a[i - k] + b[i] * c[i];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t offset = 0; offset < k; ++offset) {
            for (int64_t i = offset + k; i < len; i += k) {
                a[i] = 0.75 * a[i - k] + b[i] * c[i];
            }
        }
    }
}

