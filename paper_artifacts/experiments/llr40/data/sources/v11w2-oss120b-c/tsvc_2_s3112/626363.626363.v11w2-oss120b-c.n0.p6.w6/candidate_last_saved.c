/* Unrolled sequential implementation of TSVC tsvc_2 kernel s3112 (cumulative sum).
 * Preserves exact floating‑point behavior of the reference while reducing loop overhead.
 */

#include <stddef.h>
#include <stdint.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    int64_t i = 0;
    const int64_t stride = 4; // unroll factor
    int64_t limit = LEN_1D - (LEN_1D % stride);
    while (i < limit) {
        sum += a[i];
        b[i] = sum;
        i++;
        sum += a[i];
        b[i] = sum;
        i++;
        sum += a[i];
        b[i] = sum;
        i++;
        sum += a[i];
        b[i] = sum;
        i++;
    }
    while (i < LEN_1D) {
        sum += a[i];
        b[i] = sum;
        i++;
    }
}

