#include <stdint.h>

/* heat_3d: 7-point Laplacian stencil, Jacobi ping-pong between A and B.
 * Two half-steps per timestep: B = f(A) then A = f(B).  Interior only
 * (i,j,k in [1, N-2]); boundaries keep their initial values.
 *
 * Each output element reads six neighbours and the centre from the
 * source array and writes the destination array (restrict: no overlap),
 * so no temporaries are needed.  The k axis is contiguous in memory;
 * every stencil row is passed as its own restrict pointer so the inner
 * sweep vectorizes.  Slabs (fixed i) are independent, so they are
 * threaded with one persistent team; a barrier between the two
 * half-steps of each timestep serializes the recurrence correctly.
 */

static inline void row_step(const double *restrict rj,
                            const double *restrict rjm,
                            const double *restrict rjp,
                            const double *restrict rip,
                            const double *restrict rim,
                            double *restrict drow,
                            int64_t n, double alpha)
{
    for (int64_t k = 1; k < n - 1; ++k) {
        const double c = rj[k];
        double r = alpha * (rip[k] - 2.0 * c + rim[k]);
        r += alpha * (rjp[k] - 2.0 * c + rjm[k]);
        r += alpha * (rj[k + 1] - 2.0 * c + rj[k - 1]);
        drow[k] = r + c;
    }
}

static void slab_step(const double *restrict Sm,
                      const double *restrict Sc,
                      const double *restrict Sp,
                      double *restrict Dc,
                      int64_t N, double alpha)
{
    for (int64_t j = 1; j < N - 1; ++j) {
        row_step(Sc + j * N, Sc + (j - 1) * N, Sc + (j + 1) * N,
                 Sp + j * N, Sm + j * N, Dc + j * N, N, alpha);
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N,
                  int64_t TSTEPS, double alpha)
{
    const int64_t NN = N * N;
    #pragma omp parallel
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        #pragma omp for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i)
            slab_step(A + (i - 1) * NN, A + i * NN, A + (i + 1) * NN,
                      B + i * NN, N, alpha);
        #pragma omp for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i)
            slab_step(B + (i - 1) * NN, B + i * NN, B + (i + 1) * NN,
                      A + i * NN, N, alpha);
    }
}
