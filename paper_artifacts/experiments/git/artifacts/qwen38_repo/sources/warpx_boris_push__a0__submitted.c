/* Optimized WarpX Boris momentum pusher.
 *
 * The naive version performs ~40 separate elementwise passes over per-particle
 * temporaries (12 malloc'd arrays).  Every particle's update is a pure
 * elementwise map -- it reads only its own lane and writes only its own lane --
 * so all passes fuse into ONE loop body, the temporaries disappear, and the
 * whole particle loop is distributed across threads.  The per-element
 * arithmetic (and its association order) is unchanged, so results stay
 * numerically identical.
 */
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type, const int64_t np_particles,
                           const double q) {
    const int64_t mpt = (int64_t)(momentum_push_type);
    const double dt = 1.0e-13;
    const double econst = ((0.5 * q) * dt) / m;
    const double inv_c2 = 1.1126500560536185e-17;
    const int64_t n = np_particles;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double x = ux[i], y = uy[i], z = uz[i];
        if (mpt == 0 || mpt == 1) {
            x += econst * Ex[i];
            y += econst * Ey[i];
            z += econst * Ez[i];
        }
        double g = 1.0 / sqrt(1.0 + ((x * x + y * y) + z * z) * inv_c2);
        double txx = (econst * g) * Bx[i];
        double tyy = (econst * g) * By[i];
        double tzz = (econst * g) * Bz[i];
        if (mpt == 1 || mpt == 2) {
            const double tsq = (txx * txx) + (tyy * tyy) + (tzz * tzz);
            const double f = (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;
            txx *= f;
            tyy *= f;
            tzz *= f;
        }
        const double tsqi = 2.0 / ((1.0 + (txx * txx)) + (tyy * tyy) + (tzz * tzz));
        const double sx = txx * tsqi, sy = tyy * tsqi, sz = tzz * tsqi;
        const double xp = (x + (y * tzz)) - (z * tyy);
        const double yp = (y + (z * txx)) - (x * tzz);
        const double zp = (z + (x * tyy)) - (y * txx);
        double nx = x + (yp * sz) - (zp * sy);
        double ny = y + (zp * sx) - (xp * sz);
        double nz = z + (xp * sy) - (yp * sx);
        if (mpt == 0 || mpt == 2) {
            nx += econst * Ex[i];
            ny += econst * Ey[i];
            nz += econst * Ez[i];
        }
        ux[i] = nx;
        uy[i] = ny;
        uz[i] = nz;
    }
}
