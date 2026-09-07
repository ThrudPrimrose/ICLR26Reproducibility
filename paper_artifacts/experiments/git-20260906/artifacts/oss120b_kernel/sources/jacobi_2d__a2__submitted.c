/* Optimized Jacobi 2D stencil for double precision.
   Implements the same functionality as the reference implementation in
   /shared/tasks/jacobi_2d/jacobi_2d_reference.c but with parallelism, reduced
   overhead, and SIMD hints to improve performance.
*/

#include <stddef.h>
#include <stdint.h>

/*
 * Perform TSTEPS iterations of a 5-point Jacobi stencil on an N-by-N grid.
 * A and B are distinct arrays of size N*N; they must not overlap.
 * The function updates B from A, then A from B, for each time step.
 */
void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    const double coeff = 0.2;               // stencil coefficient
    const int64_t Nminus1 = N - 1;           // pre compute to avoid repeated subtraction

    // Assume 64‑byte alignment for SIMD; the compiler may ignore if not true.
    double *restrict A_al = __builtin_assume_aligned(A, 64);
    double *restrict B_al = __builtin_assume_aligned(B, 64);

    /* Use a single OpenMP parallel region for the entire evolution to avoid
     * repeated thread‑creation overhead, which is significant for small problem
     * sizes (preset S). Within each timestep we issue two parallel loops – one
     * updating B from A and another updating A from B. The implicit barrier
     * after each "omp for" provides the required synchronization.
     */
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            /* Update B from A */
            #pragma omp for schedule(static) nowait
            for (int64_t i = 1; i < Nminus1; ++i) {
                const int64_t row = i * N;
                const int64_t row_up = row - N;
                const int64_t row_dn = row + N;
                #pragma omp simd
                for (int64_t j = 1; j < Nminus1; ++j) {
                    const int64_t idx = row + j;
                    B_al[idx] = coeff * (
                        A_al[idx] +                 // centre
                        A_al[idx - 1] +            // left
                        A_al[idx + 1] +            // right
                        A_al[row_dn + j] +        // down (next row)
                        A_al[row_up + j]           // up (previous row)
                    );
                }
            }
            /* Update A from B */
            #pragma omp for schedule(static) nowait
            for (int64_t i = 1; i < Nminus1; ++i) {
                const int64_t row = i * N;
                const int64_t row_up = row - N;
                const int64_t row_dn = row + N;
                #pragma omp simd
                for (int64_t j = 1; j < Nminus1; ++j) {
                    const int64_t idx = row + j;
                    A_al[idx] = coeff * (
                        B_al[idx] +                 // centre
                        B_al[idx - 1] +            // left
                        B_al[idx + 1] +            // right
                        B_al[row_dn + j] +        // down
                        B_al[row_up + j]           // up
                    );
                }
            }
            /* Implicit barrier at the end of the parallel region ensures all
             * threads have completed the A‑update before the next timestep.
             */
        }
    }
}

