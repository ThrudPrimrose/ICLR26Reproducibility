/* Optimized heat_3d kernel without temporary interior buffers.
   Implements 7-point stencil using two alternating buffers A and B.
   Parallelized with OpenMP using a single parallel region.
*/
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t Nminus1 = N - 1;
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            // Compute B from A
            #pragma omp for collapse(2) schedule(static)
            for (int64_t i = 1; i < Nminus1; ++i) {
                for (int64_t j = 1; j < Nminus1; ++j) {
                    #pragma omp simd
                for (int64_t k = 1; k < Nminus1; ++k) {
                        double a_center = A[(i * N + j) * N + k];
                        double sum_neighbors =
                            A[((i + 1) * N + j) * N + k] +
                            A[((i - 1) * N + j) * N + k] +
                            A[(i * N + (j + 1)) * N + k] +
                            A[(i * N + (j - 1)) * N + k] +
                            A[(i * N + j) * N + (k + 1)] +
                            A[(i * N + j) * N + (k - 1)];
                        B[(i * N + j) * N + k] = a_center + alpha * (sum_neighbors - 6.0 * a_center);
                    }
                }
            }
            // Compute A from B
            #pragma omp for collapse(2) schedule(static)
            for (int64_t i = 1; i < Nminus1; ++i) {
                for (int64_t j = 1; j < Nminus1; ++j) {
                    #pragma omp simd
                for (int64_t k = 1; k < Nminus1; ++k) {
                        double b_center = B[(i * N + j) * N + k];
                        double sum_neighbors =
                            B[((i + 1) * N + j) * N + k] +
                            B[((i - 1) * N + j) * N + k] +
                            B[(i * N + (j + 1)) * N + k] +
                            B[(i * N + (j - 1)) * N + k] +
                            B[(i * N + j) * N + (k + 1)] +
                            B[(i * N + j) * N + (k - 1)];
                        A[(i * N + j) * N + k] = b_center + alpha * (sum_neighbors - 6.0 * b_center);
                    }
                }
            }
        }
    }
}
