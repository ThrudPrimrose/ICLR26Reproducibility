/* Optimized implementation of the scan_affine_decay kernel.
 * Computes y[i] = c[i] * y[i-1] + x[i] for i >= 1 with y[0] = x[0].
 * Parallel prefix scan using block decomposition and OpenMP.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/*
 * The function follows the C-ABI used by the benchmark suite.
 * All pointer arguments are assumed to be non-overlapping and correctly aligned.
 * LEN_1D is the length of the 1D arrays.
 */
void scan_affine_decay_fp64(double *restrict y,
                            const double *restrict c,
                            const double *restrict x,
                            const int64_t LEN_1D,
                            uint8_t *restrict workspace,
                            const int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;
    /* Seed the recurrence. */
    y[0] = x[0];
    if (LEN_1D == 1) return;

    /* Number of elements to process in the scan (indices 1 .. LEN_1D-1). */
    const int64_t n = LEN_1D - 1;
    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    int64_t chunk = (n + max_threads - 1) / max_threads; /* ceil division */

    /* Allocate per-thread reduction results. */
    double *block_A = (double *)malloc((size_t)max_threads * sizeof(double));
    double *block_B = (double *)malloc((size_t)max_threads * sizeof(double));
    if (!block_A || !block_B) {
        /* Allocation failure – fall back to serial implementation. */
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        free(block_A);
        free(block_B);
        return;
    }

    /* First pass: each thread computes the reduction (A,B) for its block.
       The reduction corresponds to composing the affine maps (c[i], x[i])
       in order: (A,B) starts as (1,0) and is updated with
       A = c[i] * A;
       B = c[i] * B + x[i]; */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = 1 + (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        double A = 1.0;
        double B = 0.0;
        for (int64_t i = start; i < end; ++i) {
            A = c[i] * A;
            B = c[i] * B + x[i];
        }
        block_A[tid] = A;
        block_B[tid] = B;
    }

    /* Compute prefix of block reductions sequentially.
       prefix_A[t] and prefix_B[t] represent the combined transformation of all
       blocks with index < t. */
    double *prefix_A = (double *)malloc((size_t)max_threads * sizeof(double));
    double *prefix_B = (double *)malloc((size_t)max_threads * sizeof(double));
    if (!prefix_A || !prefix_B) {
        free(block_A);
        free(block_B);
        free(prefix_A);
        free(prefix_B);
        /* Fall back to serial. */
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }
    prefix_A[0] = 1.0; /* identity transformation */
    prefix_B[0] = 0.0;
    for (int t = 1; t < max_threads; ++t) {
        /* Combine block (t-1) after previous prefix. */
        double Aprev = prefix_A[t - 1];
        double Bprev = prefix_B[t - 1];
        double Ablk = block_A[t - 1];
        double Bblk = block_B[t - 1];
        /* Composition: (Ablk, Bblk) ∘ (Aprev, Bprev) */
        prefix_A[t] = Ablk * Aprev;
        prefix_B[t] = Ablk * Bprev + Bblk;
    }

    /* Second pass: each thread recomputes the scan within its block, using the
       prefix transformation of all preceding blocks. */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = 1 + (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        double prefA = prefix_A[tid];
        double prefB = prefix_B[tid];
        double A_loc = 1.0;
        double B_loc = 0.0;
        double y0 = y[0];
        for (int64_t i = start; i < end; ++i) {
            A_loc = c[i] * A_loc;
            B_loc = c[i] * B_loc + x[i];
            double A_tot = prefA * A_loc;
            double B_tot = A_loc * prefB + B_loc;
            y[i] = A_tot * y0 + B_tot;
        }
    }

    free(block_A);
    free(block_B);
    free(prefix_A);
    free(prefix_B);
}

