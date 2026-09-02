/*
 * Optimized implementation of the 3-D heat diffusion stencil.
 *
 * The reference implementation in `heat_3d_reference.c` allocates two
 * temporary interior buffers (`Ac` and `Bc`) and performs the stencil using
 * these copies. The temporaries double the memory traffic and impede
 * cache utilisation. This version eliminates the temporaries entirely
 * – the stencil is expressed directly on the input arrays `A` and `B`.
 *
 * The algorithm follows the recurrence from the reference:
 *   for each time step t
 *       B = stencil(A)
 *       A = stencil(B)
 * where the stencil is the classic 7-point Laplacian with diffusion
 * coefficient `alpha` (the reference uses `alpha = 0.125`).
 *
 * By reading the centre value directly from the source array we avoid the
 * copy of the interior slab. The two updates are performed sequentially –
 * `B` is fully computed before `A` is updated, so there are no race
 * conditions.
 *
 * Parallelism:
 *   The outer two dimensions (`i` and `j`) are parallelised with OpenMP.
 *   The innermost dimension (`k`) is left as the vectorisation‑friendly loop
 *   because memory is laid out as row‑major (`k` varies fastest). Adding a
 *   `collapse(2)` clause yields a large iteration space for the runtime to
 *   distribute evenly across threads. The loops are annotated with `restrict`
 *   pointers and minimal arithmetic to aid the compiler’s auto‑vectoriser.
 */

#include <stdint.h>

/* The function name and signature are dictated by the benchmark harness.
 * `restrict` tells the compiler that the pointers do not alias, which is
 * required for safe parallelisation.
 */
void heat_3d_fp64(double *restrict A, double *restrict B,
                  int64_t N, int64_t TSTEPS, double alpha) {
    // Strides for the flattened 3‑D array layout (row‑major):
    const int64_t stride_i = N * N;   // step in the i dimension
    const int64_t stride_j = N;       // step in the j dimension
    const int64_t stride_k = 1;       // step in the k dimension (fastest)
    const int64_t Nminus1 = N - 1;

    // Main time‑step loop.
    for (int64_t t = 0; t < TSTEPS; ++t) {
        // -----------------------------------------------------------------
        // Compute B = stencil(A)
        // -----------------------------------------------------------------
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            for (int64_t j = 1; j < Nminus1; ++j) {
                // Base index for (i,j,1)
                int64_t base = i * stride_i + j * stride_j + 1;
                for (int64_t k = 1; k < Nminus1; ++k, ++base) {
                    double a_center = A[base];
                    double a_ip1 = A[base + stride_i];   // i+1
                    double a_im1 = A[base - stride_i];   // i-1
                    double a_jp1 = A[base + stride_j];   // j+1
                    double a_jm1 = A[base - stride_j];   // j-1
                    double a_kp1 = A[base + stride_k];   // k+1
                    double a_km1 = A[base - stride_k];   // k-1
                    B[base] = (alpha * (a_ip1 - 2.0 * a_center + a_im1)
                              + alpha * (a_jp1 - 2.0 * a_center + a_jm1)
                              + alpha * (a_kp1 - 2.0 * a_center + a_km1)
                              + a_center);
                }
            }
        }
        // -----------------------------------------------------------------
        // Compute A = stencil(B)
        // -----------------------------------------------------------------
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            for (int64_t j = 1; j < Nminus1; ++j) {
                int64_t base = i * stride_i + j * stride_j + 1;
                for (int64_t k = 1; k < Nminus1; ++k, ++base) {
                    double b_center = B[base];
                    double b_ip1 = B[base + stride_i];
                    double b_im1 = B[base - stride_i];
                    double b_jp1 = B[base + stride_j];
                    double b_jm1 = B[base - stride_j];
                    double b_kp1 = B[base + stride_k];
                    double b_km1 = B[base - stride_k];
                    A[base] = (alpha * (b_ip1 - 2.0 * b_center + b_im1)
                              + alpha * (b_jp1 - 2.0 * b_center + b_jm1)
                              + alpha * (b_kp1 - 2.0 * b_center + b_km1)
                              + b_center);
                }
            }
        }
    }
    // No temporaries to free.
}

