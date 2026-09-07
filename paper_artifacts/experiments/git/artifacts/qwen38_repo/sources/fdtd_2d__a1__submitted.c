/* FDTD 2D update -- optimized.
 *
 * One time step (must match the NumPy reference element by element, same op
 * order):
 *   ey[0,:]     = fict[t]
 *   ey[1:,:]   -= ey_c * (hz[1:,:]  - hz[:-1,:])     (reads OLD hz)
 *   ex[:,1:]   -= ex_c * (hz[:,1:]  - hz[:,:-1])     (reads OLD hz)
 *   hz[:-1,:-1]-= hz_c * ((ex[:-1,1:]-ex[:-1,:-1])+ey[1:,:-1]-ey[:-1,:-1])
 *                                                        (reads NEW ex, NEW ey)
 * So hz must be read everywhere before it is written, and ex/ey must be fully
 * updated before the hz pass: two parallel stages per step. The t loop is
 * serial; space is embarrassingly parallel. A persistent team covers the whole
 * t loop, so the per-step cost is two barriers, not fork/join.
 */
#include <stdint.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, const int64_t NX, const int64_t NY, const int64_t TMAX, const double ex_courant, const double ey_courant, const double hz_courant) {
    if (TMAX <= 0) return;
    if (NY <= 0 || NX <= 0) return;

#pragma omp parallel
    for (int64_t t = 0; t < TMAX; ++t) {
        const double f = fict[t];

        /* stage 1: fused ey + ex update; reads old hz only. */
#pragma omp for schedule(static)
        for (int64_t i = 0; i < NX; ++i) {
            double *const exr = ex + i * NY;
            double *const eyr = ey + i * NY;
            const double *const hzr = hz + i * NY;
            if (i == 0) {
                for (int64_t j = 0; j < NY; ++j) eyr[j] = f;
            } else {
                const double *const hzm = hz + (i - 1) * NY;
                for (int64_t j = 0; j < NY; ++j)
                    eyr[j] -= ey_courant * (hzr[j] - hzm[j]);
            }
            for (int64_t j = 1; j < NY; ++j)
                exr[j] -= ex_courant * (hzr[j] - hzr[j - 1]);
        }

        /* stage 2: hz update; reads fully-updated ex and ey. */
#pragma omp for schedule(static)
        for (int64_t i = 0; i < NX - 1; ++i) {
            const double *const exr = ex + i * NY;
            const double *const eyr = ey + i * NY;
            const double *const eyp = ey + (i + 1) * NY;
            double *const hzr = hz + i * NY;
            for (int64_t j = 0; j < NY - 1; ++j)
                hzr[j] -= hz_courant * (((exr[j + 1] - exr[j]) + eyp[j]) - eyr[j]);
        }
    }
}
