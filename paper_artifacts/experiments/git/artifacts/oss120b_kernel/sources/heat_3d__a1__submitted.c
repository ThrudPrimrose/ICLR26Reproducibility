/*
 * Optimized implementation of the heat_3d kernel.
 *
 * The original reference implementation performed four explicit interior copies
 * (A→Ac, B→Bc) and applied a series of arithmetic manipulations that expanded to a
 * long expression. The mathematical operation performed is the Jacobi update of a
 * 3‑D heat stencil:
 *
 *   new[i,j,k] = old[i,j,k] + alpha * (sum of 6 neighbours - 6 * old[i,j,k])
 *
 * where the update is applied twice per time step: first B is computed from A and
 * then A is recomputed from B. The interior copies are unnecessary – the same
 * result can be obtained by directly using the current value as the centre term.
 *
 * This implementation removes the temporary buffers, reduces arithmetic, and adds
 * straightforward OpenMP parallelisation. The loops are ordered with the k‑index
 * innermost so that memory accesses are contiguous (the layout of the 1‑D arrays
 * matches C's row‑major ordering). The code adheres to the compiler flags used by
 * the benchmark (-O3 -march=native -fopenmp -fno-math-errno -fno-trapping-math
 * -fno-signed-zeros -fstrict-aliasing -Wall -Wextra) and respects the `restrict`
 * qualifiers on the input pointers.
 */

#include <stdint.h>

/* Compute the 3‑D heat stencil using a Jacobi scheme.
 * The grid is stored in a flat array with index(i,j,k) = (i * N + j) * N + k.
 *
 * Parameters
 *   A     - pointer to the primary grid (will be overwritten each step)
 *   B     - pointer to the secondary grid (used as a temporary buffer)
 *   N     - dimension size (grid is N×N×N, N >= 3)
 *   TSTEPS- number of time steps
 *   alpha - stencil coefficient
 */
void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t Nminus1 = N - 1;            /* last valid index */
    /* pre‑compute constant (unused) */
    for (int64_t t = 0; t < TSTEPS; ++t) {
        /* Step 1: compute B from current A */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            const int64_t iN = i * N * N;
            const int64_t ipN = (i + 1) * N * N;
            const int64_t imN = (i - 1) * N * N;
            for (int64_t j = 1; j < Nminus1; ++j) {
                const int64_t ij = iN + j * N;
                const int64_t ipj = ipN + j * N;
                const int64_t imj = imN + j * N;
                const int64_t ijp = iN + (j + 1) * N;
                const int64_t ijm = iN + (j - 1) * N;
                #pragma omp simd
                    for (int64_t k = 1; k < Nminus1; ++k) {
                    // centre value
                    double a_center = A[ij + k];
                    // sum of the six neighbours
                    double sum = A[ipj + k] + A[imj + k]
                                 + A[ijp + k] + A[ijm + k]
                                 + A[ij + (k + 1)] + A[ij + (k - 1)];
                    // Jacobi update
                    B[ij + k] = a_center + alpha * (sum - 6.0 * a_center);
                }
            }
        }
        /* Step 2: compute new A from B */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 1; i < Nminus1; ++i) {
            const int64_t iN = i * N * N;
            const int64_t ipN = (i + 1) * N * N;
            const int64_t imN = (i - 1) * N * N;
            for (int64_t j = 1; j < Nminus1; ++j) {
                const int64_t ij = iN + j * N;
                const int64_t ipj = ipN + j * N;
                const int64_t imj = imN + j * N;
                const int64_t ijp = iN + (j + 1) * N;
                const int64_t ijm = iN + (j - 1) * N;
                #pragma omp simd
                    for (int64_t k = 1; k < Nminus1; ++k) {
                    double b_center = B[ij + k];
                    double sum = B[ipj + k] + B[imj + k]
                                 + B[ijp + k] + B[ijm + k]
                                 + B[ij + (k + 1)] + B[ij + (k - 1)];
                    A[ij + k] = b_center + alpha * (sum - 6.0 * b_center);
                }
            }
        }
    }
}
