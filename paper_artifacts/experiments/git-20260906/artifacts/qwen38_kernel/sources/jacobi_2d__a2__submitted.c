#include <stdint.h>

static inline void sweep_row(double *restrict dst, const double *restrict src,
                             int64_t i, int64_t N) {
    const double *up = src + (i - 1) * N;
    const double *c  = src + i * N;
    const double *dn = src + (i + 1) * N;
    double *o = dst + i * N;
    for (int64_t j = 1; j < N - 1; ++j) {
        double v = c[j];
        v += c[j - 1];
        v += c[j + 1];
        v += dn[j];
        v += up[j];
        o[j] = 0.2 * v;
    }
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
#pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
#pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i)
                sweep_row(B, A, i, N);
#pragma omp for schedule(static)
            for (int64_t i = 1; i < N - 1; ++i)
                sweep_row(A, B, i, N);
        }
    }
}
