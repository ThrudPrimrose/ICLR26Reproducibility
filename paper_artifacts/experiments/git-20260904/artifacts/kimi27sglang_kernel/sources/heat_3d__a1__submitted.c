#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t n = N;
    const int64_t n2 = n * n;
    const double a = alpha;
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                for (int64_t k = 1; k < n - 1; ++k) {
                    const double Ac = A[i * n2 + j * n + k];
                    B[i * n2 + j * n + k] =
                        a * (A[(i + 1) * n2 + j * n + k] - 2.0 * Ac + A[(i - 1) * n2 + j * n + k])
                      + a * (A[i * n2 + (j + 1) * n + k] - 2.0 * Ac + A[i * n2 + (j - 1) * n + k])
                      + a * (A[i * n2 + j * n + (k + 1)] - 2.0 * Ac + A[i * n2 + j * n + (k - 1)])
                      + Ac;
                }
            }
        }
        #pragma omp parallel for collapse(3) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                for (int64_t k = 1; k < n - 1; ++k) {
                    const double Bc = B[i * n2 + j * n + k];
                    A[i * n2 + j * n + k] =
                        a * (B[(i + 1) * n2 + j * n + k] - 2.0 * Bc + B[(i - 1) * n2 + j * n + k])
                      + a * (B[i * n2 + (j + 1) * n + k] - 2.0 * Bc + B[i * n2 + (j - 1) * n + k])
                      + a * (B[i * n2 + j * n + (k + 1)] - 2.0 * Bc + B[i * n2 + j * n + (k - 1)])
                      + Bc;
                }
            }
        }
    }
}
