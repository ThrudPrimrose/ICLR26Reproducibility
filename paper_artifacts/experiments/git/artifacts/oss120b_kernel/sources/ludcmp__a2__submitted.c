#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <math.h>

/* LU decomposition without pivoting, double precision.
   Solves A x = b using Doolittle's algorithm.
   A is overwritten with L (unit diagonal) in lower part and U in upper part.
*/

void ludcmp_fp64(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, int64_t N) {
    // LU factorization
    for (int64_t k = 0; k < N; ++k) {
        double pivot = A[k * N + k];
        // Scale column k below diagonal and apply rank-1 update.
        for (int64_t i = k + 1; i < N; ++i) {
            double *restrict row_i = A + i * N;
            double aik = row_i[k] / pivot;
            row_i[k] = aik;
            double *restrict row_k = A + k * N;
            #pragma omp simd
            for (int64_t j = k + 1; j < N; ++j) {
                row_i[j] -= aik * row_k[j];
            }
        }
    }

    // Forward substitution: solve L*y = b (L has unit diagonal)
    for (int64_t i = 0; i < N; ++i) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int64_t j = 0; j < i; ++j) {
            sum += A[i * N + j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // Backward substitution: solve U*x = y
    for (int64_t i = N; i-- > 0;) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int64_t j = i + 1; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        x[i] = (y[i] - sum) / A[i * N + i];
    }
}
