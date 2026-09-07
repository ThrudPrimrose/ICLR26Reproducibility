#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

/* Optimized heat_3d kernel: 7-point stencil in 3D.
   Uses two arrays A and B (double* restrict) and updates them alternately.
   No temporary buffers; parallelized with OpenMP.
   Parameters:
     A, B: pointers to N*N*N grid stored in row-major order (C layout).
     N: grid size in each dimension (including boundaries).
     TSTEPS: number of time steps.
     alpha: scalar coefficient for stencil.
*/
void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t Nminus1 = N - 1; // last valid index
    // Iterate over timesteps
    for (int64_t t = 0; t < TSTEPS; ++t) {
        // Update B from A (stencil)
        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            for (int64_t j = 1; j < Nminus1; ++j) {
                for (int64_t k = 1; k < Nminus1; ++k) {
                    const int64_t idx = (i * N + j) * N + k;
                    const double a_center = A[idx];
                    double sum = A[((i+1) * N + j) * N + k]   // i+1
                               + A[((i-1) * N + j) * N + k]   // i-1
                               + A[(i * N + (j+1)) * N + k]   // j+1
                               + A[(i * N + (j-1)) * N + k]   // j-1
                               + A[(i * N + j) * N + (k+1)]   // k+1
                               + A[(i * N + j) * N + (k-1)];  // k-1
                    B[idx] = a_center + alpha * (sum - 6.0 * a_center);
                }
            }
        }
        // Update A from B (stencil)
        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            for (int64_t j = 1; j < Nminus1; ++j) {
                for (int64_t k = 1; k < Nminus1; ++k) {
                    const int64_t idx = (i * N + j) * N + k;
                    const double b_center = B[idx];
                    double sum = B[((i+1) * N + j) * N + k]
                               + B[((i-1) * N + j) * N + k]
                               + B[(i * N + (j+1)) * N + k]
                               + B[(i * N + (j-1)) * N + k]
                               + B[(i * N + j) * N + (k+1)]
                               + B[(i * N + j) * N + (k-1)];
                    A[idx] = b_center + alpha * (sum - 6.0 * b_center);
                }
            }
        }
    }
}
