#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

void covariance_fp64(double *restrict cov, double *restrict data, int64_t M, int64_t N, double float_n) {
    // Compute column means
    double *mean = (double *)malloc((size_t)M * sizeof(double));
    if (!mean) return;
    // Initialize mean to zero
    for (int64_t i = 0; i < M; ++i) mean[i] = 0.0;
    // Compute sums for each column
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int64_t k = 0; k < N; ++k) {
            sum += data[k * M + i];
        }
        mean[i] = sum / (double)N;
    }
    // Subtract mean from each element (center data)
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < N; ++k) {
        double *row = &data[k * M];
        for (int64_t i = 0; i < M; ++i) {
            row[i] -= mean[i];
        }
    }
    // Initialize output covariance matrix to zero (upper triangle only)
    for (int64_t i = 0; i < M * M; ++i) cov[i] = 0.0;
    // Compute covariance via blocked outer-product accumulation
    const int64_t B = 64; // block size (tunable)
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i0 = 0; i0 < M; i0 += B) {
        for (int64_t j0 = i0; j0 < M; j0 += B) {
            int64_t i_max = i0 + B;
            if (i_max > M) i_max = M;
            int64_t j_max = j0 + B;
            if (j_max > M) j_max = M;
            int64_t bi = i_max - i0;
            int64_t bj = j_max - j0;
            // Allocate a local block buffer initialized to zero
            double *block = (double *)calloc((size_t)bi * (size_t)bj, sizeof(double));
            if (!block) continue; // skip on allocation failure
            // Accumulate contributions from each sample
            for (int64_t k = 0; k < N; ++k) {
                double *row = &data[k * M];
                for (int64_t i = i0; i < i_max; ++i) {
                    double vi = row[i];
                    for (int64_t j = j0; j < j_max; ++j) {
                        block[(i - i0) * bj + (j - j0)] += vi * row[j];
                    }
                }
            }
            // Write block back to the global covariance matrix
            for (int64_t i = i0; i < i_max; ++i) {
                for (int64_t j = j0; j < j_max; ++j) {
                    cov[i * M + j] += block[(i - i0) * bj + (j - j0)];
                }
            }
            free(block);
        }
    }
    // Mirror upper triangle to lower triangle
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = i + 1; j < M; ++j) {
            cov[j * M + i] = cov[i * M + j];
        }
    }
    // Scale by (float_n - 1.0)
    double denom = float_n - 1.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M * M; ++i) {
        cov[i] /= denom;
    }
    free(mean);
}
