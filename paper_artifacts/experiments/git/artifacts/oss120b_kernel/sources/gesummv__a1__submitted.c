/* Optimized implementation of GESUMMV kernel
 * Computes out = alpha * A @ x + beta * B @ x
 * where A and B are N x N matrices, x is a vector of length N.
 * This version avoids temporary matrices and uses OpenMP for parallelism.
 */
#include <stddef.h>
#include <stdint.h>

void gesummv_fp64(const double *restrict A, const double *restrict B,
                  double *restrict out, const double *restrict x,
                  int64_t N, double alpha, double beta) {
    // Parallelize over rows of the output vector.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double sumA = 0.0;
        double sumB = 0.0;
        const double *rowA = A + i * N;
        const double *rowB = B + i * N;
        // Compute alpha * A[i,:] * x and beta * B[i,:] * x.
        for (int64_t j = 0; j < N; ++j) {
            double xj = x[j];
            sumA += (alpha * rowA[j]) * xj;
            sumB += (beta * rowB[j]) * xj;
        }
        out[i] = sumA + sumB;
    }
}
