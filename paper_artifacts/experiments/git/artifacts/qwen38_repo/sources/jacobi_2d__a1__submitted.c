#include <stdint.h>
void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS) {
    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N-1; ++i) {
            const double *restrict Am = A + (i-1)*N;
            const double *restrict Ab = A + (i  )*N;
            const double *restrict Ap = A + (i+1)*N;
            double *restrict Bb = B + i*N;
            for (int64_t j = 1; j < N-1; ++j)
                Bb[j] = 0.2 * ((((Ab[j] + Ab[j-1]) + Ab[j+1]) + Ap[j]) + Am[j]);
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N-1; ++i) {
            const double *restrict Bm = B + (i-1)*N;
            const double *restrict Bb = B + (i  )*N;
            const double *restrict Bp = B + (i+1)*N;
            double *restrict Ab = A + i*N;
            for (int64_t j = 1; j < N-1; ++j)
                Ab[j] = 0.2 * ((((Bb[j] + Bb[j-1]) + Bb[j+1]) + Bp[j]) + Bm[j]);
        }
    }
}
