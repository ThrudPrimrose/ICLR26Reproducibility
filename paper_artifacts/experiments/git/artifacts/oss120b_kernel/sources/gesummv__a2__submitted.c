/*
 * Optimized implementation of the ``gesummv`` kernel for the HPCAgent-Bench.
 *
 * The kernel computes:
 *   out = alpha * (A @ x) + beta * (B @ x)
 * where ``A`` and ``B`` are dense square matrices of size ``N x N`` and ``x``
 * is a vector of length ``N``. All arguments are double‑precision values.
 *
 * The reference implementation (generated from the NumPy reference) allocates
 * temporary scaled copies of ``A`` and ``B`` before performing the matrix‑vector
 * products. That incurs a large memory footprint and two additional traversals
 * of the matrices, which severely limits performance.
 *
 * This version avoids any dynamic allocation and fuses the scaling with the
 * matrix‑vector multiplication. The algorithm follows the reference exactly —
 * it multiplies each matrix element by the corresponding scalar (``alpha`` or
 * ``beta``) *before* the dot‑product, thereby matching the rounding behaviour
 * of the reference implementation.
 *
 * Parallelism:
 *   * The outer loop over rows is parallelised with OpenMP. Each iteration works
 *     on independent data, so no synchronisation is required.
 *   * The inner loop is vectorised using ``#pragma omp simd`` with reductions on
 *     the two partial sums. This gives the compiler explicit information that
 *     the loop is safe to vectorise.
 *
 * The function signature follows the convention of other kernels (e.g.
 * ``covariance_fp64``) and matches the reference exactly:
 *
 *   void gesummv_fp64(const double *restrict A,
 *                     const double *restrict B,
 *                     double *restrict out,
 *                     const double *restrict x,
 *                     int64_t N,
 *                     double alpha,
 *                     double beta);
 */

#include <stdint.h>
#include <omp.h>

void gesummv_fp64(const double *restrict A,
                  const double *restrict B,
                  double *restrict out,
                  const double *restrict x,
                  int64_t N,
                  double alpha,
                  double beta) {
    /* Outer loop over rows – safe to parallelise because each row writes a
     * distinct element of ``out`` and only reads from the input arrays. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const double *restrict rowA = A + i * N;
        const double *restrict rowB = B + i * N;
        double sumA = 0.0;
        double sumB = 0.0;
        /* The inner product is vectorisable. Using a combined reduction for the
         * two sums lets the compiler generate a single SIMD loop with two
         * accumulators. */
        #pragma omp simd reduction(+:sumA,sumB)
        for (int64_t j = 0; j < N; ++j) {
            sumA += (alpha * rowA[j]) * x[j];
            sumB += (beta * rowB[j]) * x[j];
        }
        out[i] = sumA + sumB;
    }
}

