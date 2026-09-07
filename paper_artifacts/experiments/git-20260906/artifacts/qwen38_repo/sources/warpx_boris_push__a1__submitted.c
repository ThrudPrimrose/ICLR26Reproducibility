// Optimized WarpX Boris momentum pusher.
//
// The naive implementation ran ~45 separate single-pass loops over the particle
// arrays with 18 temporary buffers (12-18 per particle of malloc traffic per
// call).  Every particle only touches its own lanes, so the whole update is a
// pure elementwise map: it is fused into one loop kept entirely in registers,
// parallelized with OpenMP, and vectorized (AVX-512 with -march=native).
//
// The per-particle operation order is kept identical to the reference
// (line-for-line from UpdateMomentumBoris), so results are numerically
// unchanged.  Only the loop structure / memory traffic / parallelism changed.

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

constexpr double dt = 1e-13;

/* Per-particle Boris update with the three flag-combinations spelled out so the
 * branch on momentum_push_type happens once, outside the hot loop. */

static inline void body_full(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                             const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                             double *restrict ux, double *restrict uy, double *restrict uz,
                             double econst, double inv_c2, int64_t i)
{
    double x = ux[i], y = uy[i], z = uz[i];
    x += (econst * Ex[i]);
    y += (econst * Ey[i]);
    z += (econst * Ez[i]);

    double g = sqrt((1.0 + ((((x * x) + (y * y)) + (z * z)) * inv_c2)));
    double ig = (1.0 / g);
    double a = (econst * ig);
    double tx = (a * Bx[i]);
    double ty = (a * By[i]);
    double tz = (a * Bz[i]);

    double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
    double sx = (tx * tsqi);
    double sy = (ty * tsqi);
    double sz = (tz * tsqi);

    double x_p = ((x + (y * tz)) - (z * ty));
    double y_p = ((y + (z * tx)) - (x * tz));
    double z_p = ((z + (x * ty)) - (y * tx));

    double x1 = (x + ((y_p * sz) - (z_p * sy)));
    double y1 = (y + ((z_p * sx) - (x_p * sz)));
    double z1 = (z + ((x_p * sy) - (y_p * sx)));

    x1 += (econst * Ex[i]);
    y1 += (econst * Ey[i]);
    z1 += (econst * Ez[i]);

    ux[i] = x1;
    uy[i] = y1;
    uz[i] = z1;
}

static inline void body_first(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                              const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                              double *restrict ux, double *restrict uy, double *restrict uz,
                              double econst, double inv_c2, int64_t i)
{
    double x = ux[i], y = uy[i], z = uz[i];
    x += (econst * Ex[i]);
    y += (econst * Ey[i]);
    z += (econst * Ez[i]);

    double g = sqrt((1.0 + ((((x * x) + (y * y)) + (z * z)) * inv_c2)));
    double ig = (1.0 / g);
    double a = (econst * ig);
    double tx = (a * Bx[i]);
    double ty = (a * By[i]);
    double tz = (a * Bz[i]);

    double tsq = (((tx * tx) + (ty * ty)) + (tz * tz));
    bool has_field = (tsq > 0.0);
    double safe_tsq = (has_field ? tsq : 1.0);
    double factor = (has_field ? ((sqrt((1.0 + tsq)) - 1.0) / safe_tsq) : 0.5);
    tx *= factor;
    ty *= factor;
    tz *= factor;

    double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
    double sx = (tx * tsqi);
    double sy = (ty * tsqi);
    double sz = (tz * tsqi);

    double x_p = ((x + (y * tz)) - (z * ty));
    double y_p = ((y + (z * tx)) - (x * tz));
    double z_p = ((z + (x * ty)) - (y * tx));

    ux[i] = (x + ((y_p * sz) - (z_p * sy)));
    uy[i] = (y + ((z_p * sx) - (x_p * sz)));
    uz[i] = (z + ((x_p * sy) - (y_p * sx)));
}

static inline void body_second(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                               const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                               double *restrict ux, double *restrict uy, double *restrict uz,
                               double econst, double inv_c2, int64_t i)
{
    double x = ux[i], y = uy[i], z = uz[i];

    double g = sqrt((1.0 + ((((x * x) + (y * y)) + (z * z)) * inv_c2)));
    double ig = (1.0 / g);
    double a = (econst * ig);
    double tx = (a * Bx[i]);
    double ty = (a * By[i]);
    double tz = (a * Bz[i]);

    double tsq = (((tx * tx) + (ty * ty)) + (tz * tz));
    bool has_field = (tsq > 0.0);
    double safe_tsq = (has_field ? tsq : 1.0);
    double factor = (has_field ? ((sqrt((1.0 + tsq)) - 1.0) / safe_tsq) : 0.5);
    tx *= factor;
    ty *= factor;
    tz *= factor;

    double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
    double sx = (tx * tsqi);
    double sy = (ty * tsqi);
    double sz = (tz * tsqi);

    double x_p = ((x + (y * tz)) - (z * ty));
    double y_p = ((y + (z * tx)) - (x * tz));
    double z_p = ((z + (x * ty)) - (y * tx));

    double x1 = (x + ((y_p * sz) - (z_p * sy)));
    double y1 = (y + ((z_p * sx) - (x_p * sz)));
    double z1 = (z + ((x_p * sy) - (y_p * sx)));

    x1 += (econst * Ex[i]);
    y1 += (econst * Ey[i]);
    z1 += (econst * Ez[i]);

    ux[i] = x1;
    uy[i] = y1;
    uz[i] = z1;
}

static inline void body_none(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                             double *restrict ux, double *restrict uy, double *restrict uz,
                             double econst, double inv_c2, int64_t i)
{
    double x = ux[i], y = uy[i], z = uz[i];

    double g = sqrt((1.0 + ((((x * x) + (y * y)) + (z * z)) * inv_c2)));
    double ig = (1.0 / g);
    double a = (econst * ig);
    double tx = (a * Bx[i]);
    double ty = (a * By[i]);
    double tz = (a * Bz[i]);

    double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
    double sx = (tx * tsqi);
    double sy = (ty * tsqi);
    double sz = (tz * tsqi);

    double x_p = ((x + (y * tz)) - (z * ty));
    double y_p = ((y + (z * tx)) - (x * tz));
    double z_p = ((z + (x * ty)) - (y * tx));

    ux[i] = (x + ((y_p * sz) - (z_p * sy)));
    uy[i] = (y + ((z_p * sx) - (x_p * sz)));
    uz[i] = (z + ((x_p * sy) - (y_p * sx)));
}

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type, const int64_t np_particles,
                           const double q)
{
    const int64_t mpt = ((int64_t)(momentum_push_type));
    const double econst = (((0.5 * q) * dt) / m);
    const double inv_c2 = 1.1126500560536185e-17;
    const int64_t n = np_particles;

    if (mpt == 1) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i)
            body_first(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, i);
    } else if (mpt == 2) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i)
            body_second(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, i);
    } else if (mpt == 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i)
            body_full(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, i);
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i)
            body_none(Bx, By, Bz, ux, uy, uz, econst, inv_c2, i);
    }
}
