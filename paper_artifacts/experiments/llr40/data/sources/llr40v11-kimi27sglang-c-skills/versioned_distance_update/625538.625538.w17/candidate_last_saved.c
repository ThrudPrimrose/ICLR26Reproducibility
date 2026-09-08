#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#ifndef VL
#define VL 8
#endif

static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

static inline int64_t next_pow2_i64(int64_t x) {
    int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

void versioned_distance_update_fp64(double *restrict a, double *restrict b, double *restrict c,
                                    int64_t K, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    if (K <= 0 || LEN_1D <= K) return;

    if (K == 1) {
        double x = a[0];
        int64_t i = 1;
        for (; i + 3 < LEN_1D; i += 4) {
            x = 0.75 * x + b[i] * c[i];     a[i] = x;
            x = 0.75 * x + b[i+1] * c[i+1]; a[i+1] = x;
            x = 0.75 * x + b[i+2] * c[i+2]; a[i+2] = x;
            x = 0.75 * x + b[i+3] * c[i+3]; a[i+3] = x;
        }
        for (; i < LEN_1D; ++i) {
            x = 0.75 * x + b[i] * c[i];
            a[i] = x;
        }
        return;
    }

    int64_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;

    // For large K: split each K-block into d contiguous tiles and process
    // the d independent "chains" of tiles in parallel.  Each chain computes
    // the same sub-tile position of successive K-blocks, so within a chain
    // tiles are dependent, but chains are independent.  Choose d as a
    // power of two >= threads so every core is busy.
    if (K >= 256 && (K & (K - 1)) == 0) {
        int64_t d = next_pow2_i64(nthreads);
        if (d > K / VL) d = K / VL;
        if (d < 1) d = 1;
        int64_t T = K / d;                 // tile size, divides K
        if (T >= VL) {
            int64_t NT = (LEN_1D + T - 1) / T;
#pragma omp parallel for schedule(static, 1)
            for (int64_t p = 0; p < d; ++p) {
                for (int64_t t = p; t < NT; t += d) {
                    int64_t start = t * T;
                    if (start < K) start = K;
                    int64_t end = (t + 1) * T;
                    if (end > LEN_1D) end = LEN_1D;
#pragma omp simd
                    for (int64_t i = start; i < end; ++i) {
                        a[i] = 0.75 * a[i - K] + b[i] * c[i];
                    }
                }
            }
            return;
        }
    }

    // Moderate K: compiler auto-vectorises across consecutive iterations.
    for (int64_t i = K; i < LEN_1D; ++i) {
        a[i] = 0.75 * a[i - K] + b[i] * c[i];
    }
}
