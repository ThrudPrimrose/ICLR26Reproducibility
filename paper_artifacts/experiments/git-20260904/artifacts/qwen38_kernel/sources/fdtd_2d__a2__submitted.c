/* FDTD 2D update -- ascending-row scheme with per-block pre-phase.
 *
 * Per timestep the reference does 3 grid-wide passes (ey, ex, hz) reading
 * hz 3 times.  Here each thread owns a contiguous block of rows and, after
 * a short pre-phase (which may only read hz, nothing is destroyed yet),
 * streams its rows in ascending order:
 *
 *   P0: thread t computes new ey[r0] from old hz (still intact).
 *   P1: for i = r0..r1-1:
 *         (a) if i < r1-1: new ey[i+1] = old ey[i+1] - cey*(old hz[i+1] - old hz[i])
 *         (b) new ex[i]   -= cex*(old hz[i] - old hz[i-1])         (j>=1)
 *         (c) if i < NX-1: new hz[i]   -= chz*((new ex[j+1]-new ex[j])
 *                                                + new ey[i+1][j] - new ey[i][j])
 *
 * Row r1 (the block's last row) skips (a): its successor row's ey was already
 * stored by the next thread in P0, and P1 only ever reads old hz of its own
 * block rows, so no cross-thread hazard remains (one barrier per timestep).
 * Every per-element expression keeps the reference's exact operation order,
 * so results are bit-identical.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant)
{
    for (int64_t t = 0; t < TMAX; ++t) {
        const double fv = fict[t];
        #pragma omp parallel
        {
            const int nt  = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const int64_t per = NX / nt;
            const int64_t rem = NX % nt;
            const int64_t r0 = per * tid + (tid < rem ? tid : rem);
            const int64_t r1 = r0 + per + (tid < rem ? 1 : 0);

            if (r0 < r1) {
                /* P0: new ey for the block's first row (old hz still intact) */
                double *e = ey + (size_t)r0 * (size_t)NY;
                if (r0 == 0) {
                    #pragma omp simd
                    for (int64_t j = 0; j < NY; ++j) e[j] = fv;
                } else {
                    const double *h0 = hz + (size_t)r0 * (size_t)NY;
                    const double *h1 = hz + (size_t)(r0 - 1) * (size_t)NY;
                    #pragma omp simd
                    for (int64_t j = 0; j < NY; ++j)
                        e[j] -= ey_courant * (h0[j] - h1[j]);
                }
            }
            #pragma omp barrier
            for (int64_t i = r0; i < r1; ++i) {
                double *h = hz + (size_t)i * (size_t)NY;
                double *x = ex + (size_t)i * (size_t)NY;
                if (i < r1 - 1) {
                    double *e   = ey + (size_t)(i + 1) * (size_t)NY;
                    const double *h0 = hz + (size_t)(i + 1) * (size_t)NY;
                    #pragma omp simd
                    for (int64_t j = 0; j < NY; ++j)
                        e[j] -= ey_courant * (h0[j] - h[j]);
                }
                #pragma omp simd
                for (int64_t j = 1; j < NY; ++j)
                    x[j] -= ex_courant * (h[j] - h[j - 1]);
                if (i < NX - 1) {
                    const double *e1 = ey + (size_t)i * (size_t)NY;
                    const double *e2 = ey + (size_t)(i + 1) * (size_t)NY;
                    #pragma omp simd
                    for (int64_t j = 0; j < NY - 1; ++j)
                        h[j] -= hz_courant * ((x[j + 1] - x[j]) + e2[j] - e1[j]);
                }
            }
        }
    }
}
