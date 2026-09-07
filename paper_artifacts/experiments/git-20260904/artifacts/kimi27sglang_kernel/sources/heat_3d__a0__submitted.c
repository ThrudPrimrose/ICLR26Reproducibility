#include <stdint.h>

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha)
{
    const int64_t n = N;

    for (int64_t t = 1; t <= TSTEPS; ++t) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                const int64_t base_ij  = (i * n + j) * n;
                const int64_t base_im1 = ((i - 1) * n + j) * n;
                const int64_t base_ip1 = ((i + 1) * n + j) * n;
                const int64_t base_jm1 = (i * n + (j - 1)) * n;
                const int64_t base_jp1 = (i * n + (j + 1)) * n;

                #pragma omp simd
                for (int64_t k = 1; k < n - 1; ++k) {
                    double ac = A[base_ij + k];
                    double sum = A[base_ip1 + k]
                               + A[base_im1 + k]
                               + A[base_jp1 + k]
                               + A[base_jm1 + k]
                               + A[base_ij + k + 1]
                               + A[base_ij + k - 1];
                    B[base_ij + k] = alpha * (sum - 6.0 * ac) + ac;
                }
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < n - 1; ++i) {
            for (int64_t j = 1; j < n - 1; ++j) {
                const int64_t base_ij  = (i * n + j) * n;
                const int64_t base_im1 = ((i - 1) * n + j) * n;
                const int64_t base_ip1 = ((i + 1) * n + j) * n;
                const int64_t base_jm1 = (i * n + (j - 1)) * n;
                const int64_t base_jp1 = (i * n + (j + 1)) * n;

                #pragma omp simd
                for (int64_t k = 1; k < n - 1; ++k) {
                    double bc = B[base_ij + k];
                    double sum = B[base_ip1 + k]
                               + B[base_im1 + k]
                               + B[base_jp1 + k]
                               + B[base_jm1 + k]
                               + B[base_ij + k + 1]
                               + B[base_ij + k - 1];
                    A[base_ij + k] = alpha * (sum - 6.0 * bc) + bc;
                }
            }
        }
    }
}
