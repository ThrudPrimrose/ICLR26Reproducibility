#include <stdint.h>
#include <stdlib.h>

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t N2 = N * N;
    const double a2 = -2.0;
    for (int64_t t = 0; t < TSTEPS; ++t) {
        /* A -> B */
        #pragma omp parallel for schedule(dynamic, 4)
        for (int64_t i = 1; i < N - 1; ++i) {
            for (int64_t j = 1; j < N - 1; ++j) {
                const double *rA = A + (i * N + j) * N;
                double *rB = B + (i * N + j) * N;
                for (int64_t k = 1; k < N - 1; ++k) {
                    double ac = rA[k];
                    rB[k] = alpha * (rA[k - 1] + a2 * ac + rA[k + 1])
                           + alpha * (rA[k - N] + a2 * ac + rA[k + N])
                           + alpha * (rA[k - N2] + a2 * ac + rA[k + N2]) + ac;
                }
            }
        }
        /* B -> A */
        #pragma omp parallel for schedule(dynamic, 4)
        for (int64_t i = 1; i < N - 1; ++i) {
            for (int64_t j = 1; j < N - 1; ++j) {
                const double *rB = B + (i * N + j) * N;
                double *rA = A + (i * N + j) * N;
                for (int64_t k = 1; k < N - 1; ++k) {
                    double bc = rB[k];
                    rA[k] = alpha * (rB[k - 1] + a2 * bc + rB[k + 1])
                           + alpha * (rB[k - N] + a2 * bc + rB[k + N])
                           + alpha * (rB[k - N2] + a2 * bc + rB[k + N2]) + bc;
                }
            }
        }
    }
}
