#include <stdint.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            #pragma omp simd
            for (int64_t j = 1; j < N - 1; ++j) {
                B[i*N + j] = 0.2 * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (j+1)] + A[(i+1)*N + j] + A[(i-1)*N + j]);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            #pragma omp simd
            for (int64_t j = 1; j < N - 1; ++j) {
                A[i*N + j] = 0.2 * (B[i*N + j] + B[i*N + (j-1)] + B[i*N + (j+1)] + B[(i+1)*N + j] + B[(i-1)*N + j]);
            }
        }
    }
}
