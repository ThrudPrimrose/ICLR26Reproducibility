#include <stdint.h>
#include <stddef.h>

// LU decomposition without pivoting, and forward/back substitution.
// Implements the same algorithm as the reference `ludcmp_fp64` but avoids temporary storage
// and uses OpenMP to parallelize the rank‑1 update of the trailing submatrix.
// The function signature follows the C‑ABI expected by the benchmark.

void ludcmp_fp64(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, int64_t N) {
    // LU factorization (Doolittle, no pivoting)
    for (int64_t k = 0; k < N; ++k) {
        double pivot = A[k * N + k];
        // Scale column below diagonal and update trailing submatrix.
        // Parallelize across rows i > k.
        #pragma omp parallel for schedule(static)
        for (int64_t i = k + 1; i < N; ++i) {
            double L = A[i * N + k] / pivot; // multiplier
            A[i * N + k] = L;                // store L in place of A[i,k]
            // Rank‑1 update: A[i, j] -= L * A[k, j] for j>k
            #pragma omp simd
            for (int64_t j = k + 1; j < N; ++j) {
                A[i * N + j] -= L * A[k * N + j];
            }
        }
    }

    // Forward substitution: solve L*y = b (L has unit diagonal).
    for (int64_t i = 0; i < N; ++i) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int64_t j = 0; j < i; ++j) {
            sum += A[i * N + j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // Back substitution: solve U*x = y.
    for (int64_t i = N - 1; i >= 0; --i) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int64_t j = i + 1; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        x[i] = (y[i] - sum) / A[i * N + i];
    }
}
