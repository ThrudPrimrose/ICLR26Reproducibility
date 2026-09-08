#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 4096
#endif

void versioned_distance_update_fp64(double *restrict a, const double *restrict b, const double *restrict c, int64_t K, int64_t LEN_1D) {
    if (K <= 0 || K >= LEN_1D) return;

    const double alpha = 0.75;

    if (K == 1) {
        // Specialised K=1: blocked parallel scan using the output array as scratch for d.
        if (LEN_1D < 2048) {
            for (int64_t i = 1; i < LEN_1D; ++i) {
                a[i] = alpha * a[i - 1] + b[i] * c[i];
            }
            return;
        }

        const int64_t B = BLOCK;
        int64_t num_blocks = (LEN_1D - 1 + B - 1) / B; // indices 1..N-1
        double *agg = (double *)aligned_alloc(64, 2 * num_blocks * sizeof(double));
        if (!agg) return; // fail-safe, though should not happen
        double *agg_p = agg;
        double *agg_q = agg + num_blocks;

        #pragma omp parallel for schedule(static)
        for (int64_t t = 0; t < num_blocks; ++t) {
            int64_t start = 1 + t * B;
            int64_t end = start + B;
            if (end > LEN_1D) end = LEN_1D;
            double acc = 0.0;
            double p = 1.0;
            for (int64_t i = start; i < end; ++i) {
                double d = b[i] * c[i];
                a[i] = d;
                acc = alpha * acc + d;
                p *= alpha;
            }
            agg_p[t] = p;
            agg_q[t] = acc;
        }

        double *carry = (double *)aligned_alloc(64, num_blocks * sizeof(double));
        if (!carry) { free(agg); return; }
        double cur = a[0];
        for (int64_t t = 0; t < num_blocks; ++t) {
            carry[t] = cur;
            cur = agg_p[t] * cur + agg_q[t];
        }
        free(agg);

        #pragma omp parallel for schedule(static)
        for (int64_t t = 0; t < num_blocks; ++t) {
            int64_t start = 1 + t * B;
            int64_t end = start + B;
            if (end > LEN_1D) end = LEN_1D;
            double v = carry[t];
            for (int64_t i = start; i < end; ++i) {
                v = alpha * v + a[i];
                a[i] = v;
            }
        }
        free(carry);
        return;
    }

    // Generic K > 1: each residue class modulo K forms an independent chain.
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < K; ++r) {
        int64_t i = r + K;
        for (; i < LEN_1D; i += K) {
            a[i] = alpha * a[i - K] + b[i] * c[i];
        }
    }
}
