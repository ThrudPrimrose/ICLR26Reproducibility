/* Optimized heat_3d kernel using direct stencil without interior buffers.
 * Parallelized with OpenMP and explicit SIMD for inner loop.
 */
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    if (N <= 2 || TSTEPS <= 0) return;
    const int64_t N2 = N * N;          // N squared
    // Assume pointers are 64-byte aligned for SIMD
    double *restrict A_aligned = __builtin_assume_aligned(A, 64);
    double *restrict B_aligned = __builtin_assume_aligned(B, 64);
    for (int64_t t = 0; t < TSTEPS; ++t) {
        // Compute B from A
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            int64_t iN = i * N;
            for (int64_t j = 1; j < N - 1; ++j) {
                int64_t inner = N - 2;
                int64_t base = (iN + j) * N + 1; // start of inner region
                #pragma omp simd aligned(A_aligned, B_aligned:64)
                for (int64_t k = 0; k < inner; ++k) {
                    int64_t idx = base + k;
                    double a_center = A_aligned[idx];
                    double sum =
                        A_aligned[idx + 1] + A_aligned[idx - 1] +
                        A_aligned[idx + N] + A_aligned[idx - N] +
                        A_aligned[idx + N2] + A_aligned[idx - N2];
                    B_aligned[idx] = a_center + alpha * (sum - 6.0 * a_center);
                }
            }
        }
        // Compute A from B
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            int64_t iN = i * N;
            for (int64_t j = 1; j < N - 1; ++j) {
                int64_t inner = N - 2;
                int64_t base = (iN + j) * N + 1;
                #pragma omp simd aligned(A_aligned, B_aligned:64)
                for (int64_t k = 0; k < inner; ++k) {
                    int64_t idx = base + k;
                    double b_center = B_aligned[idx];
                    double sum =
                        B_aligned[idx + 1] + B_aligned[idx - 1] +
                        B_aligned[idx + N] + B_aligned[idx - N] +
                        B_aligned[idx + N2] + B_aligned[idx - N2];
                    A_aligned[idx] = b_center + alpha * (sum - 6.0 * b_center);
                }
            }
        }
    }
}
