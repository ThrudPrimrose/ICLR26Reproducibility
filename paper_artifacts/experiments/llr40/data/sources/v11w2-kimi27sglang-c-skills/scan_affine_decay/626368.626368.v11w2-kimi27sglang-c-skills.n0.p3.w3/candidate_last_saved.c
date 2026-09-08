#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x, double *restrict y, const int64_t LEN_1D) {
    const int64_t N = LEN_1D;
    if (N <= 0) return;
    if (N <= 4096) {
        y[0] = x[0];
        for (int64_t i = 1; i < N; i++) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    const int num_threads = omp_get_max_threads();
    int64_t block_size = N / (4 * num_threads);
    if (block_size < 4096) block_size = 4096;
    int64_t M = (N + block_size - 1) / block_size;
    if (M < num_threads) M = num_threads;
    block_size = (N + M - 1) / M;

    double *A = (double *)malloc((size_t)M * sizeof(double));
    double *B = (double *)malloc((size_t)M * sizeof(double));
    double *prefix_A = (double *)malloc((size_t)M * sizeof(double));
    double *prefix_B = (double *)malloc((size_t)M * sizeof(double));
    if (!A || !B || !prefix_A || !prefix_B) {
        y[0] = x[0];
        for (int64_t i = 1; i < N; i++) y[i] = c[i] * y[i - 1] + x[i];
        free(A); free(B); free(prefix_A); free(prefix_B);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < M; k++) {
        int64_t l = k * block_size;
        int64_t r = l + block_size;
        if (r > N) r = N;
        double a = 1.0;
        double b = 0.0;
        for (int64_t i = r - 1; i >= l; i--) {
            b = b + a * x[i];
            a = a * c[i];
        }
        A[k] = a;
        B[k] = b;
    }

    prefix_A[0] = A[0];
    prefix_B[0] = B[0];
    for (int64_t k = 1; k < M; k++) {
        prefix_A[k] = A[k] * prefix_A[k - 1];
        prefix_B[k] = A[k] * prefix_B[k - 1] + B[k];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < M; k++) {
        int64_t l = k * block_size;
        int64_t r = l + block_size;
        if (r > N) r = N;
        double prev = (k == 0) ? 0.0 : prefix_B[k - 1];
        for (int64_t i = l; i < r; i++) {
            y[i] = c[i] * prev + x[i];
            prev = y[i];
        }
    }

    free(A);
    free(B);
    free(prefix_A);
    free(prefix_B);
}
