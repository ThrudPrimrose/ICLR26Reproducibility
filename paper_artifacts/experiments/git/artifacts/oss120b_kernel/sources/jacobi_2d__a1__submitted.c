/* Optimized implementation of the Jacobi 2-D stencil kernel */
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B,
                    int64_t N, int64_t TSTEPS) {
    const double coeff = 0.2;
    const int64_t n = N;
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            int64_t row = i * n;
            int64_t row_up = (i - 1) * n;
            int64_t row_down = (i + 1) * n;
            #pragma omp simd
            for (int64_t j = 1; j < n - 1; ++j) {
                B[row + j] = coeff * (A[row + j] + A[row + j - 1] + A[row + j + 1] + A[row_up + j] + A[row_down + j]);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            int64_t row = i * n;
            int64_t row_up = (i - 1) * n;
            int64_t row_down = (i + 1) * n;
            #pragma omp simd
            for (int64_t j = 1; j < n - 1; ++j) {
                A[row + j] = coeff * (B[row + j] + B[row + j - 1] + B[row + j + 1] + B[row_up + j] + B[row_down + j]);
            }
        }
    }
}
