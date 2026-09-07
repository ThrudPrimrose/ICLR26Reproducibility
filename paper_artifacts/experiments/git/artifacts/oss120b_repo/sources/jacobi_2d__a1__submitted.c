// Optimized Jacobi 2D stencil with explicit OpenMP parallel loops and SIMD
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <omp.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS) {
    const double coeff = 0.2;
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            // Compute B from A
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i) {
                #pragma omp simd
                for (int64_t j = 1; j < N - 1; ++j) {
                    const int64_t idx = i * N + j;
                    B[idx] = coeff * (A[idx] + A[idx - 1] + A[idx + 1] + A[idx - N] + A[idx + N]);
                }
            }
            // Compute A from B
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i) {
                #pragma omp simd
                for (int64_t j = 1; j < N - 1; ++j) {
                    const int64_t idx = i * N + j;
                    A[idx] = coeff * (B[idx] + B[idx - 1] + B[idx + 1] + B[idx - N] + B[idx + N]);
                }
            }
        }
    }
}
