/* Optimized Jacobi 2D stencil using OpenMP parallelism and vectorization.
   Original reference implementation:
   void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS)
   This version adds OpenMP `for` parallelism over the outer dimension and
   `#pragma omp simd` on the inner dimension to encourage SIMD vectorisation.
*/

#include <stdint.h>
#include <omp.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    const double c0 = 0.2;
    // Parallel region outermost to avoid creating/destroying threads each timestep.
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            // Update B from A
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N-1; ++i) {
                #pragma omp simd
                for (int64_t j = 1; j < N-1; ++j) {
                    B[i*N + j] = c0 * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (j+1)] + A[(i+1)*N + j] + A[(i-1)*N + j]);
                }
            }
            // Update A from B
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N-1; ++i) {
                #pragma omp simd
                for (int64_t j = 1; j < N-1; ++j) {
                    A[i*N + j] = c0 * (B[i*N + j] + B[i*N + (j-1)] + B[i*N + (j+1)] + B[(i+1)*N + j] + B[(i-1)*N + j]);
                }
            }
        }
    }
}
