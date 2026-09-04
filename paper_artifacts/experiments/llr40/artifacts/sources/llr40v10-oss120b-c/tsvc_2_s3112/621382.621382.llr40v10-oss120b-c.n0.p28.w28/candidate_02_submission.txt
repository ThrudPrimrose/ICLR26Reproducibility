/* Optimized TSVC tsvc_2 s3112 kernel with loop unrolling (factor 4) and alignment hints.
 * Computes prefix sum of input array a into output array b exactly as reference.
 */

#include <stdint.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    const double *restrict aptr = (const double *restrict) __builtin_assume_aligned(a, 64);
    double *restrict bptr = (double *restrict) __builtin_assume_aligned(b, 64);
    double sum = 0.0;
    int64_t i = 0;
    int64_t n = LEN_1D;
    const int64_t UNROLL = 4;
    int64_t limit = n - (n % UNROLL);
    for (; i < limit; i += UNROLL) {
        sum += aptr[0]; bptr[0] = sum;
        sum += aptr[1]; bptr[1] = sum;
        sum += aptr[2]; bptr[2] = sum;
        sum += aptr[3]; bptr[3] = sum;
        aptr += UNROLL;
        bptr += UNROLL;
    }
    for (; i < n; ++i) {
        sum += *aptr++;
        *bptr++ = sum;
    }
}

