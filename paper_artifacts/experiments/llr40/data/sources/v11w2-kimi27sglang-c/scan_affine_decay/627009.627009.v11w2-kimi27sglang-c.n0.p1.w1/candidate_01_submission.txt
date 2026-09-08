#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

static inline int64_t ceil_div64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline int64_t round_up64(int64_t a, int64_t b) {
    return ceil_div64(a, b) * b;
}

void scan_affine_decay_fp64(double* __restrict__ c,
                            double* __restrict__ x,
                            double* __restrict__ y,
                            int64_t LEN_1D,
                            uint8_t* __restrict__ workspace,
                            int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0) return;

    // Sequential fast path for small inputs.
    if (LEN_1D < 4096) {
        y[0] = x[0];
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    int64_t nthreads = omp_get_max_threads();
    int64_t nblocks = nthreads * 8;
    int64_t block = ceil_div64(LEN_1D, nblocks);

    size_t alloc_bytes = round_up64(nblocks * (int64_t)sizeof(double), 64);
    double* block_C = (double*)aligned_alloc(64, alloc_bytes);
    double* block_X = (double*)aligned_alloc(64, alloc_bytes);

    // Phase 1: each block computes its local scan and its aggregate affine map
    // (C_b, X_b) such that actual y[end-1] = C_b * y[start-1] + X_b.
    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t blocks_per_thread = ceil_div64(nblocks, nt);
        int64_t b_start = (int64_t)tid * blocks_per_thread;
        int64_t b_end = b_start + blocks_per_thread;
        if (b_end > nblocks) b_end = nblocks;

        for (int64_t b = b_start; b < b_end; ++b) {
            int64_t start = b * block;
            int64_t end = start + block;
            if (end > LEN_1D) end = LEN_1D;

            y[start] = x[start];
            double C = c[start];
            for (int64_t i = start + 1; i < end; ++i) {
                y[i] = c[i] * y[i - 1] + x[i];
                C *= c[i];
            }
            block_C[b] = C;
            block_X[b] = y[end - 1];
        }
    }

    // Phase 2: serial scan over the block-level affine maps.
    double prefix = 0.0;
    for (int64_t b = 0; b < nblocks; ++b) {
        double new_prefix = block_C[b] * prefix + block_X[b];
        block_X[b] = prefix;       // prefix input to block b
        prefix = new_prefix;
    }

    // Phase 3: apply the block prefix to every element inside each block.
    // actual y[i] = local_y[i] + prefix_b * product(c[start..i]).
    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t blocks_per_thread = ceil_div64(nblocks, nt);
        int64_t b_start = (int64_t)tid * blocks_per_thread;
        int64_t b_end = b_start + blocks_per_thread;
        if (b_end > nblocks) b_end = nblocks;

        for (int64_t b = b_start; b < b_end; ++b) {
            if (b == 0) continue;  // block 0 has prefix 0, nothing to add
            int64_t start = b * block;
            int64_t end = start + block;
            if (end > LEN_1D) end = LEN_1D;
            double P = block_X[b];
            double prod = 1.0;
            for (int64_t i = start; i < end; ++i) {
                prod *= c[i];
                y[i] += P * prod;
            }
        }
    }

    free(block_C);
    free(block_X);
}
