#include <stdint.h>

static void sweep64(const double *restrict S, double *restrict D, int64_t N, double alpha, double beta) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 1; i <= N - 2; ++i)
        for (int64_t j = 1; j <= N - 2; ++j) {
            const double *const C = S + (i * N + j) * N + 1;
            const double *const P = C + N * N;
            const double *const M = C - N * N;
            const double *const R = C + N;
            const double *const L = C - N;
            double *const O = D + (i * N + j) * N + 1;
            const int64_t n = N - 2;
            for (int64_t k = 0; k < n; ++k)
                O[k] = alpha * (C[k + 1] + C[k - 1] + P[k] + M[k] + R[k] + L[k]) + beta * C[k];
        }
}

static void sweep32(const float *restrict S, float *restrict D, int64_t N, float alpha, float beta) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 1; i <= N - 2; ++i)
        for (int64_t j = 1; j <= N - 2; ++j) {
            const float *const C = S + (i * N + j) * N + 1;
            const float *const P = C + N * N;
            const float *const M = C - N * N;
            const float *const R = C + N;
            const float *const L = C - N;
            float *const O = D + (i * N + j) * N + 1;
            const int64_t n = N - 2;
            for (int64_t k = 0; k < n; ++k)
                O[k] = alpha * (C[k + 1] + C[k - 1] + P[k] + M[k] + R[k] + L[k]) + beta * C[k];
        }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const double beta = 1.0 - 6.0 * alpha;
    if (N < 3 || TSTEPS < 1) return;
    #pragma omp parallel
    {
        for (int64_t t = 1; t <= TSTEPS; ++t) {
            sweep64(A, B, N, alpha, beta);
            sweep64(B, A, N, alpha, beta);
        }
    }
}

void heat_3d_fp32(float *restrict A, float *restrict B, int64_t N, int64_t TSTEPS, float alpha) {
    const float beta = 1.0f - 6.0f * alpha;
    if (N < 3 || TSTEPS < 1) return;
    #pragma omp parallel
    {
        for (int64_t t = 1; t <= TSTEPS; ++t) {
            sweep32(A, B, N, alpha, beta);
            sweep32(B, A, N, alpha, beta);
        }
    }
}
