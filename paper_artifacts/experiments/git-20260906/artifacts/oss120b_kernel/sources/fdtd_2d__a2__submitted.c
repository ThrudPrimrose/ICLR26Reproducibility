/* Optimized FDtd 2D kernel for double precision.
   Original reference in /shared/tasks/fdtd_2d/fdtd_2d_reference.c.
   Merged updates of ey and ex to reduce parallel loop overhead.
*/

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict,
                  double *restrict hz, int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant) {
    const int nx = (int)NX;
    const int ny = (int)NY;
    const int tmax = (int)TMAX;
    #pragma omp parallel
    {
        for (int t = 0; t < tmax; ++t) {
            // Set first row of ey to fictitious source value
            #pragma omp for nowait
            for (int si1 = 0; si1 < ny; ++si1) {
                ey[si1] = fict[t];
            }

            // Update ey rows (except first) and ex columns together
            #pragma omp for nowait
            for (int si0 = 0; si0 < nx; ++si0) {
                double *cur_ey = ey + (size_t)si0 * ny;
                const double *cur_hz = hz + (size_t)si0 * ny;
                const double *prev_hz = (si0 > 0) ? hz + (size_t)(si0 - 1) * ny : NULL;
                double *ex_row = ex + (size_t)si0 * ny;
                const double *hz_row = cur_hz;
                if (si0 > 0) {
                    #pragma omp simd
                    for (int si1 = 0; si1 < ny; ++si1) {
                        cur_ey[si1] -= ey_courant * (cur_hz[si1] - prev_hz[si1]);
                    }
                }
                #pragma omp simd
                for (int si1 = 1; si1 < ny; ++si1) {
                    ex_row[si1] -= ex_courant * (hz_row[si1] - hz_row[si1 - 1]);
                }
            }
            // Ensure ey and ex updates are complete before hz update
            #pragma omp barrier

            // Update hz for interior cells
            #pragma omp for
            for (int si0 = 0; si0 < nx - 1; ++si0) {
                double *hz_row = hz + (size_t)si0 * ny;
                const double *ex_row = ex + (size_t)si0 * ny;
                const double *ey_row = ey + (size_t)si0 * ny;
                const double *ey_next = ey + (size_t)(si0 + 1) * ny;
                #pragma omp simd
                for (int si1 = 0; si1 < ny - 1; ++si1) {
                    hz_row[si1] -= hz_courant * ((ex_row[si1 + 1] - ex_row[si1]) + (ey_next[si1] - ey_row[si1]));
                }
            }
        }
    }
}
