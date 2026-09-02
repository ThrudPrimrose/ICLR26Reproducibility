// Optimized heat_3d kernel
#include <stdint.h>
#include <omp.h>

// Compute the 3D heat equation stencil over TSTEPS time steps.
// A and B are N x N x N arrays stored in row-major order (k varies fastest).
// The boundaries (i=0,N-1; j=0,N-1; k=0,N-1) are unchanged.

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t stride_i = N * N; // offset for i +/- 1
    const int64_t stride_j = N;     // offset for j +/- 1
    const int64_t stride_k = 1;     // offset for k +/- 1
    // Main time stepping loop.
    for (int64_t t = 0; t < TSTEPS; ++t) {
        // ---- Update B from A ----
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            for (int64_t j = 1; j < N - 1; ++j) {
                int64_t base = (i * N + j) * N; // index of (i,j,0)
                #pragma omp simd
                for (int64_t k = 1; k < N - 1; ++k) {
                    int64_t idx = base + k;
                    double center = A[idx];
                    double sum = A[idx + stride_i]   // i+1
                               + A[idx - stride_i]   // i-1
                               + A[idx + stride_j]   // j+1
                               + A[idx - stride_j]   // j-1
                               + A[idx + stride_k]   // k+1
                               + A[idx - stride_k];  // k-1
                    B[idx] = alpha * (sum - 6.0 * center) + center;
                }
            }
        }
        // ---- Update A from B ----
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            for (int64_t j = 1; j < N - 1; ++j) {
                int64_t base = (i * N + j) * N;
                #pragma omp simd
                for (int64_t k = 1; k < N - 1; ++k) {
                    int64_t idx = base + k;
                    double center = B[idx];
                    double sum = B[idx + stride_i]
                               + B[idx - stride_i]
                               + B[idx + stride_j]
                               + B[idx - stride_j]
                               + B[idx + stride_k]
                               + B[idx - stride_k];
                    A[idx] = alpha * (sum - 6.0 * center) + center;
                }
            }
        }
    }
}
