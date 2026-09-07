#include <stdint.h>
#include <omp.h>

/* Optimized Jacobi 2D stencil for double precision.
 * A and B are N x N arrays stored in row-major order.
 * The kernel performs TSTEPS timesteps, updating interior points.
 * The outer boundary is left unchanged.
 */
void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    // Assume 64-byte alignment for best SIMD performance.
    double *restrict A_al = (double *restrict)__builtin_assume_aligned(A, 64);
    double *restrict B_al = (double *restrict)__builtin_assume_aligned(B, 64);
    const double coeff = 0.2; // stencil coefficient
    const int64_t n = N;
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            // Compute B = stencil(A_al)
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < n - 1; ++i) {
                double *restrict a_row      = A_al + i * n;
                double *restrict a_row_up   = A_al + (i - 1) * n;
                double *restrict a_row_down = A_al + (i + 1) * n;
                double *restrict b_row      = B_al + i * n;
                #pragma omp simd
                for (int64_t j = 1; j < n - 1; ++j) {
                    b_row[j] = coeff * (a_row[j] + a_row[j - 1] + a_row[j + 1] + a_row_up[j] + a_row_down[j]);
                }
            }
            // Compute A = stencil(B_al)
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < n - 1; ++i) {
                double *restrict b_row      = B_al + i * n;
                double *restrict b_row_up   = B_al + (i - 1) * n;
                double *restrict b_row_down = B_al + (i + 1) * n;
                double *restrict a_row      = A_al + i * n;
                #pragma omp simd
                for (int64_t j = 1; j < n - 1; ++j) {
                    a_row[j] = coeff * (b_row[j] + b_row[j - 1] + b_row[j + 1] + b_row_up[j] + b_row_down[j]);
                }
            }
        }
    }
}
