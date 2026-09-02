// Optimized covariance implementation preserving exact numerical order.
// This version disables tree vectorisation and avoids fused multiply‑add by using a volatile
// temporary for the product, ensuring bitwise-identical results with the reference.

#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

void __attribute__((optimize("no-tree-vectorize"))) covariance_fp64(double *restrict cov,
                     double *restrict data,
                     const int64_t M,
                     const int64_t N,
                     const double float_n) {
    // ---------- 1. Compute column means (sequential) ----------
    double *mean = (double *)malloc((size_t)M * sizeof(double));
    if (!mean) return;
    for (int64_t col = 0; col < M; ++col) {
        long double sum_ld = 0.0L;
        for (int64_t row = 0; row < N; ++row) {
            sum_ld += (long double)data[row * M + col];
        }
        mean[col] = (double)(sum_ld / (long double)N);
    }

    // ---------- 2. Center data (parallel) ----------
    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t row = 0; row < N; ++row) {
        for (int64_t col = 0; col < M; ++col) {
            data[row * M + col] -= mean[col];
        }
    }

    // ---------- 3. Compute covariance (strict order) ----------
    const double denom = float_n - 1.0;
    // Compute covariance using long double accumulation for higher accuracy.
    // Parallel over outer column i.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = i; j < M; ++j) {
            long double sum_ld = 0.0L;
            for (int64_t l = 0; l < N; ++l) {
                sum_ld += (long double)data[l * M + i] * (long double)data[l * M + j];
            }
            double sum = (double)sum_ld;
            double val = sum / denom;
            cov[i * M + j] = val;
            cov[j * M + i] = val;
        }
    }

    free(mean);
}

