/* Optimized Jacobi 2D stencil kernel
 * Implements the same functionality as the reference `jacobi_2d_fp64`.
 * Optimizations applied:
 *   - OpenMP parallel region reused across timesteps.
 *   - Parallel for over rows with static scheduling.
 *   - SIMD vectorization of inner loop using `#pragma omp simd`.
 *   - Pointer arithmetic to avoid repeated index calculations.
 *   - `restrict` qualifiers on input pointers.
 *   - `__builtin_assume_aligned` to hint alignment (64-byte for AVX-512).
 */

#include <stddef.h>
#include <stdint.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    const double factor = 0.2;
    const int64_t n = N;
    // Assume 64-byte alignment for best SIMD performance.
    A = (double *)__builtin_assume_aligned(A, 64);
    B = (double *)__builtin_assume_aligned(B, 64);

    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            // Compute B = stencil(A)
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < n - 1; ++i) {
                double *restrict a_row      = A + i * n;
                double *restrict a_row_up   = A + (i - 1) * n;
                double *restrict a_row_down = A + (i + 1) * n;
                double *restrict b_row      = B + i * n;
                #pragma omp simd
                for (int64_t j = 1; j < n - 1; ++j) {
                    b_row[j] = factor * (a_row[j] + a_row[j - 1] + a_row[j + 1] + a_row_up[j] + a_row_down[j]);
                }
            }
            // Compute A = stencil(B)
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < n - 1; ++i) {
                double *restrict b_row      = B + i * n;
                double *restrict b_row_up   = B + (i - 1) * n;
                double *restrict b_row_down = B + (i + 1) * n;
                double *restrict a_row      = A + i * n;
                #pragma omp simd
                for (int64_t j = 1; j < n - 1; ++j) {
                    a_row[j] = factor * (b_row[j] + b_row[j - 1] + b_row[j + 1] + b_row_up[j] + b_row_down[j]);
                }
            }
        }
    }
}

