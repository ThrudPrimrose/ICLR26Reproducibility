// out = alpha * (A @ x) + beta * (B @ x)   (BLAS gesummv, dense row-major)
//
// Optimized relative to the naive generated code:
//   * no temporary matrices: rows of A and B are consumed directly, so the
//     kernel streams 16*N*N bytes instead of the ~48*N*N the copy-then-gemv
//     version moved;
//   * each row is two dot products accumulated in 8 independent partial
//     sums per matrix, which lets the compiler (-O3 -march=native) emit
//     wide SIMD FMA chains (256/512-bit) instead of a serial scalar
//     accumulate;
//   * rows are independent, so the parallel region splits the row index
//     range across threads; within a row the partial sums are combined in
//     a fixed tree, so results do not depend on thread count.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <omp.h>
#include <stddef.h>

void gesummv_fp64(const double *restrict A, const double *restrict B, double *restrict out, const double *restrict x, const int64_t N, const double alpha, const double beta) {
    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t i_begin = (N * tid) / nt;
        const int64_t i_end = (N * (tid + 1)) / nt;
        for (int64_t i = i_begin; i < i_end; ++i) {
            const double *a = A + (size_t)i * (size_t)N;
            const double *b = B + (size_t)i * (size_t)N;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            double a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
            double b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            double b4 = 0.0, b5 = 0.0, b6 = 0.0, b7 = 0.0;
            int64_t j = 0;
            for (; j + 8 <= N; j += 8) {
                a0 += a[j] * x[j];
                a1 += a[j + 1] * x[j + 1];
                a2 += a[j + 2] * x[j + 2];
                a3 += a[j + 3] * x[j + 3];
                a4 += a[j + 4] * x[j + 4];
                a5 += a[j + 5] * x[j + 5];
                a6 += a[j + 6] * x[j + 6];
                a7 += a[j + 7] * x[j + 7];
                b0 += b[j] * x[j];
                b1 += b[j + 1] * x[j + 1];
                b2 += b[j + 2] * x[j + 2];
                b3 += b[j + 3] * x[j + 3];
                b4 += b[j + 4] * x[j + 4];
                b5 += b[j + 5] * x[j + 5];
                b6 += b[j + 6] * x[j + 6];
                b7 += b[j + 7] * x[j + 7];
            }
            double da = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            double db = ((b0 + b1) + (b2 + b3)) + ((b4 + b5) + (b6 + b7));
            for (; j < N; ++j) {
                da += a[j] * x[j];
                db += b[j] * x[j];
            }
            out[i] = alpha * da + beta * db;
        }
    }
}
