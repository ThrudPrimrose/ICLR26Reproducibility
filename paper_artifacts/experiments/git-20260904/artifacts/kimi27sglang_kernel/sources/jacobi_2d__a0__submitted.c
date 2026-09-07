#include <stdint.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    const int64_t n = N;
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            double *restrict a_row = A + i * n;
            double *restrict b_row = B + i * n;
            const double *restrict a_prev = A + (i - 1) * n;
            const double *restrict a_next = A + (i + 1) * n;
            #pragma omp simd
            for (int64_t j = 1; j < n - 1; ++j) {
                b_row[j] = 0.2 * ((((a_row[j] + a_row[j - 1]) + a_row[j + 1]) + a_next[j]) + a_prev[j]);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            double *restrict b_row = B + i * n;
            double *restrict a_row = A + i * n;
            const double *restrict b_prev = B + (i - 1) * n;
            const double *restrict b_next = B + (i + 1) * n;
            #pragma omp simd
            for (int64_t j = 1; j < n - 1; ++j) {
                a_row[j] = 0.2 * ((((b_row[j] + b_row[j - 1]) + b_row[j + 1]) + b_next[j]) + b_prev[j]);
            }
        }
    }
}
