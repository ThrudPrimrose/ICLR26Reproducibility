// Optimized heat_3d implementation
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

// Original kernel symbol: heat_3d_fp64
void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS, const double alpha) {
    // Precompute some constants for indexing
    const int64_t Nm1 = N - 1;
    const size_t NN = (size_t)N * (size_t)N; // N*N
    // Main time-stepping loop
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        // Compute B interior from A
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nm1; ++i) {
            for (int64_t j = 1; j < Nm1; ++j) {
                size_t idx_base = (size_t)i * NN + (size_t)j * (size_t)N + 1; // k starts at 1
                #pragma omp simd
                for (int64_t k = 1; k < Nm1; ++k) {
                    size_t idx = idx_base + (size_t)k - 1; // compute flat index
                    double center = A[idx];
                    double neighbor_i = A[idx + NN] - 2.0 * center + A[idx - NN]; // i-direction (x) neighbors
                    double neighbor_j = A[idx + N] - 2.0 * center + A[idx - N]; // j-direction (y) neighbors
                    double neighbor_k = A[idx + 1] - 2.0 * center + A[idx - 1]; // k-direction (z) neighbors
                    B[idx] = alpha * (neighbor_i + neighbor_j + neighbor_k) + center;
                }
            }
        }
        // Compute A interior from B
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nm1; ++i) {
            for (int64_t j = 1; j < Nm1; ++j) {
                size_t idx_base = (size_t)i * NN + (size_t)j * (size_t)N + 1;
                #pragma omp simd
                for (int64_t k = 1; k < Nm1; ++k) {
                    size_t idx = idx_base + (size_t)k - 1;
                    double center = B[idx];
                    double neighbor_i = B[idx + NN] - 2.0 * center + B[idx - NN]; // i-direction (x) neighbors
                    double neighbor_j = B[idx + N] - 2.0 * center + B[idx - N]; // j-direction (y) neighbors
                    double neighbor_k = B[idx + 1] - 2.0 * center + B[idx - 1]; // k-direction (z) neighbors
                    A[idx] = alpha * (neighbor_i + neighbor_j + neighbor_k) + center;
                }
            }
        }
    }
}
