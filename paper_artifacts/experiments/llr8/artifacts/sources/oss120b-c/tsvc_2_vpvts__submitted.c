/*
 * TSVC_2 kernel "vpvts" reference implementation in C.
 *
 * The kernel computes a saxpy-like operation:
 *    a[i] = a[i] + b[i] * S
 * for i in [0, LEN_1D). This matches the Python NumPy reference.
 *
 * The function is declared with `restrict` to aid alias analysis, and uses
 * OpenMP for parallelism and SIMD vectorisation.
 *
 * Signature expected by the benchmark harness:
 *   void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D, int64_t S, uint8_t *restrict workspace, int64_t workspace_bytes);
 */

#include <stddef.h>
#include <stdint.h>
#include <omp.h>

/* Perform: a[i] += b[i] * S for i = 0 .. LEN_1D-1 */
void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D, int64_t S, uint8_t *restrict workspace, int64_t workspace_bytes) {
    // Hint alignment to the compiler (64-byte) if possible.
    a = (double *) __builtin_assume_aligned(a, 64);
    b = (const double *) __builtin_assume_aligned(b, 64);
    (void)workspace; (void)workspace_bytes;
    const double dS = (double) S;
    /* Parallelise across threads and vectorise the inner operation. */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] + b[i] * dS;
    }
}

