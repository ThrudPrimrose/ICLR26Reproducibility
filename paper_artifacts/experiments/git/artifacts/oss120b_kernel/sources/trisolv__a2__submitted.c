// Optimized forward substitution for lower triangular solve (naive vectorized)
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void trisolv_fp64(const double *restrict L, const double *restrict b, double *restrict x, int64_t N) {
    // Solve forward substitution: L * x = b, where L is lower triangular.
    // This implementation uses a simple O(N^2) algorithm with SIMD vectorization of the inner dot product.
    for (int64_t i = 0; i < N; ++i) {
        const double *row = L + i * N; // pointer to the i-th row of L
        double sum = 0.0;
        // Compute dot product of row[0..i-1] with x[0..i-1]
        //#pragma omp simd reduction(+:sum)
        for (int64_t j = 0; j < i; ++j) {
            sum += row[j] * x[j];
        }
        // Compute x[i] = (b[i] - sum) / L[i,i]
        x[i] = (b[i] - sum) / row[i];
    }
}
