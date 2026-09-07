// Optimized 2-D Jacobi stencil (5-point, double buffering A<->B).
// Two half-sweeps per timestep; the timestep loop is a serial recurrence,
// so parallelism is over the interior rows (each writes one distinct row).
// Inner loops vectorize with -march=native (AVX2/AVX-512 as available).
#include <stdint.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS) {
    if (N < 3 || TSTEPS <= 0) return;

#pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
#pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i) {
                const double *restrict a  = A + i * N;
                const double *restrict am = A + (i - 1) * N;
                const double *restrict ap = A + (i + 1) * N;
                double *restrict b = B + i * N;
                for (int64_t j = 1; j < N - 1; ++j) {
                    b[j] = 0.2 * ((((a[j] + a[j - 1]) + a[j + 1]) + ap[j]) + am[j]);
                }
            }
#pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i) {
                const double *restrict b  = B + i * N;
                const double *restrict bm = B + (i - 1) * N;
                const double *restrict bp = B + (i + 1) * N;
                double *restrict a = A + i * N;
                for (int64_t j = 1; j < N - 1; ++j) {
                    a[j] = 0.2 * ((((b[j] + b[j - 1]) + b[j + 1]) + bp[j]) + bm[j]);
                }
            }
        }
    }
}
