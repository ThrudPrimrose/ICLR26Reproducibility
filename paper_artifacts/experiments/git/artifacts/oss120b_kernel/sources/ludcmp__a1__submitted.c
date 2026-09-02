#include <stdint.h>
#include <omp.h>

/*
 * Optimized LU decomposition without pivoting.
 * Input:
 *   - A: NxN matrix stored in row-major order; will be overwritten with L and U factors (L has unit diagonal implicit).
 *   - b: right-hand side vector of length N.
 *   - x: output solution vector (size N).
 *   - y: temporary workspace vector of length N (used for forward substitution).
 *   - N: matrix dimension.
 *
 * This implementation follows the standard Doolittle algorithm with a rank-1 update
 * performed directly, eliminating the temporary VLA used in the reference version.
 * The loops are written to enable the compiler to vectorize the inner j-loop and to
 * improve cache reuse of the pivot row A[k,:].
 */

void ludcmp_fp64(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, int64_t N) {
    // LU factorization (Doolittle) – no row pivoting.
    for (int64_t k = 0; k < N; ++k) {
        double pivot = A[k * N + k];
        // Compute multipliers L[i][k] and update trailing sub-matrix.
        for (int i = k + 1; i < N; ++i) {
            double lik = A[i * N + k] / pivot;
            A[i * N + k] = lik; // store multiplier (lower triangular element)
            double *row_i = &A[i * N];
            const double *row_k = &A[k * N];
            #pragma omp simd
                for (int j = k + 1; j < N; ++j) {
                row_i[j] -= lik * row_k[j];
            }
        }
    }

    // Forward substitution: solve Ly = b.
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        const double *row_i = &A[i * N];
        #pragma omp simd reduction(+:sum)
            for (int j = 0; j < i; ++j) {
            sum += row_i[j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // Back substitution: solve Ux = y.
    for (int i = (int)N - 1; i >= 0; --i) {
        double sum = 0.0;
        const double *row_i = &A[i * N];
        #pragma omp simd reduction(+:sum)
            for (int j = i + 1; j < N; ++j) {
            sum += row_i[j] * x[j];
        }
        x[i] = (y[i] - sum) / row_i[i];
    }
}
