// Optimized heat_3d_fp64: in-place 7-point stencil, no auxiliary copy buffers.
#include <stdint.h>
#include <omp.h>

void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS, const double alpha) {
    const int64_t n = N;
    const int64_t nn = n * n;

    #pragma omp parallel
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        #pragma omp for collapse(2) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                #pragma omp simd
                for (int64_t k = 1; k < n - 1; ++k) {
                    const int64_t idx = i * nn + j * n + k;
                    const double c = A[idx];
                    B[idx] = (((alpha * (A[idx + nn] - 2.0 * c + A[idx - nn]))
                             + alpha * (A[idx + n]  - 2.0 * c + A[idx - n]))
                             + alpha * (A[idx + 1]  - 2.0 * c + A[idx - 1]))
                             + c;
                }
            }
        }

        #pragma omp for collapse(2) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                #pragma omp simd
                for (int64_t k = 1; k < n - 1; ++k) {
                    const int64_t idx = i * nn + j * n + k;
                    const double c = B[idx];
                    A[idx] = (((alpha * (B[idx + nn] - 2.0 * c + B[idx - nn]))
                             + alpha * (B[idx + n]  - 2.0 * c + B[idx - n]))
                             + alpha * (B[idx + 1]  - 2.0 * c + B[idx - 1]))
                             + c;
                }
            }
        }
    }
}
