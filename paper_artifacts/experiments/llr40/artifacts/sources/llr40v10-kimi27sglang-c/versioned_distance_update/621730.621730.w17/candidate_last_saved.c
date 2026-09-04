#include <stdint.h>

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t LEN_1D,
                                    const int64_t K) {
    if (K <= 0 || K >= LEN_1D) {
        return;
    }

    if (K == 1) {
        double carry = a[0];
        for (int64_t i = 1; i < LEN_1D; ++i) {
            carry = 0.75 * carry + b[i] * c[i];
            a[i] = carry;
        }
        return;
    }

    /* Small/medium K: process independent blocks of size K sequentially.
       All K elements inside a block depend only on the previous block, so
       the inner block vectorises cleanly with contiguous memory accesses. */
    if (K <= 512) {
        for (int64_t base = K; base < LEN_1D; base += K) {
            int64_t end = base + K;
            if (end > LEN_1D) end = LEN_1D;
            #pragma omp simd
            for (int64_t i = base; i < end; ++i) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
        return;
    }

    /* Large K: each block has enough work to pay for a parallel region.
       We keep one persistent parallel region and synchronise between blocks. */
    #pragma omp parallel
    {
        for (int64_t base = K; base < LEN_1D; base += K) {
            int64_t end = base + K;
            if (end > LEN_1D) end = LEN_1D;
            #pragma omp for simd schedule(static)
            for (int64_t i = base; i < end; ++i) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
    }
}
