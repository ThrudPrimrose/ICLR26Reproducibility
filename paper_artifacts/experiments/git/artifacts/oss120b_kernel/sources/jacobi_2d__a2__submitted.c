/* Optimized C kernel for jacobi_2d benchmark */
#include <stdint.h>
#include <omp.h>

/* Jacobi 2D stencil, double precision.
   A and B are NxN arrays stored in row-major order.
   N >= 3. The interior points (1..N-2) are updated.
   TSTEPS is the number of time steps.
   The routine updates B from A, then A from B for each step.
*/
void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    const double alpha = 0.2;
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            /* Update B from A */
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N-1; ++i) {
                double *restrict Arow = A + i * N;
                double *restrict Arow_up = A + (i-1) * N;
                double *restrict Arow_down = A + (i+1) * N;
                double *restrict Brow = B + i * N;
                #pragma omp simd
                for (int64_t j = 1; j < N-1; ++j) {
                    Brow[j] = alpha * (Arow[j] + Arow[j-1] + Arow[j+1] + Arow_up[j] + Arow_down[j]);
                }
            }
            /* Update A from B */
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < N-1; ++i) {
                double *restrict Brow = B + i * N;
                double *restrict Brow_up = B + (i-1) * N;
                double *restrict Brow_down = B + (i+1) * N;
                double *restrict Arow = A + i * N;
                #pragma omp simd
                for (int64_t j = 1; j < N-1; ++j) {
                    Arow[j] = alpha * (Brow[j] + Brow[j-1] + Brow[j+1] + Brow_up[j] + Brow_down[j]);
                }
            }
        }
    }
}
