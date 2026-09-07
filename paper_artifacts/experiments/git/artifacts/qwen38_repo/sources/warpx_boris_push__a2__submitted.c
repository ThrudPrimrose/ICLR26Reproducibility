/* Optimized WarpX Boris momentum pusher.
 *
 * The naive implementation streamed 12+ temporary arrays through memory for
 * what is a purely elementwise, carry-free per-particle computation.  This
 * version fuses the whole per-particle math into one register-resident
 * sequence (the exact same arithmetic, in the same order, as the reference)
 * and parallelizes the particle loop across cores with OpenMP.  The
 * momentum_push_type branch is uniform for the whole call, so it is resolved
 * outside the loop into three clean, fully auto-vectorizable variants.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

static const double DT = 1e-13;
static const double INV_C2 = 1.1126500560536185e-17; /* 1.0/(299792458.0*299792458.0) */

static void boris_full(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                       const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                       double *restrict ux, double *restrict uy, double *restrict uz,
                       int64_t n, double econst)
{
    const double inv_c2 = INV_C2;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        /* first half-push for E */
        double u0 = ux[i] + econst * Ex[i];
        double u1 = uy[i] + econst * Ey[i];
        double u2 = uz[i] + econst * Ez[i];

        /* relativistic gamma */
        double r = (u0 * u0 + u1 * u1) + u2 * u2;
        double invg = 1.0 / sqrt(1.0 + r * inv_c2);

        /* magnetic rotation t vector */
        double k = econst * invg;
        double tx = k * Bx[i];
        double ty = k * By[i];
        double tz = k * Bz[i];

        double s = 2.0 / (1.0 + (tx * tx + ty * ty) + tz * tz);
        double sx = tx * s, sy = ty * s, sz = tz * s;

        double u0p = u0 + u1 * tz - u2 * ty;
        double u1p = u1 + u2 * tx - u0 * tz;
        double u2p = u2 + u0 * ty - u1 * tx;

        u0 += u1p * sz - u2p * sy;
        u1 += u2p * sx - u0p * sz;
        u2 += u0p * sy - u1p * sx;

        /* second half-push for E */
        u0 += econst * Ex[i];
        u1 += econst * Ey[i];
        u2 += econst * Ez[i];

        ux[i] = u0;
        uy[i] = u1;
        uz[i] = u2;
    }
}

static void boris_first(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                        const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                        double *restrict ux, double *restrict uy, double *restrict uz,
                        int64_t n, double econst)
{
    const double inv_c2 = INV_C2;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double u0 = ux[i] + econst * Ex[i];
        double u1 = uy[i] + econst * Ey[i];
        double u2 = uz[i] + econst * Ez[i];

        double r = (u0 * u0 + u1 * u1) + u2 * u2;
        double invg = 1.0 / sqrt(1.0 + r * inv_c2);

        double k = econst * invg;
        double tx = k * Bx[i];
        double ty = k * By[i];
        double tz = k * Bz[i];

        /* half-push rescale of t */
        double tsq = (tx * tx + ty * ty) + tz * tz;
        double f = (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;
        tx *= f;
        ty *= f;
        tz *= f;

        double s = 2.0 / (1.0 + (tx * tx + ty * ty) + tz * tz);
        double sx = tx * s, sy = ty * s, sz = tz * s;

        double u0p = u0 + u1 * tz - u2 * ty;
        double u1p = u1 + u2 * tx - u0 * tz;
        double u2p = u2 + u0 * ty - u1 * tx;

        u0 += u1p * sz - u2p * sy;
        u1 += u2p * sx - u0p * sz;
        u2 += u0p * sy - u1p * sx;

        ux[i] = u0;
        uy[i] = u1;
        uz[i] = u2;
    }
}

static void boris_second(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                         const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                         double *restrict ux, double *restrict uy, double *restrict uz,
                         int64_t n, double econst)
{
    const double inv_c2 = INV_C2;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double u0 = ux[i];
        double u1 = uy[i];
        double u2 = uz[i];

        double r = (u0 * u0 + u1 * u1) + u2 * u2;
        double invg = 1.0 / sqrt(1.0 + r * inv_c2);

        double k = econst * invg;
        double tx = k * Bx[i];
        double ty = k * By[i];
        double tz = k * Bz[i];

        double tsq = (tx * tx + ty * ty) + tz * tz;
        double f = (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;
        tx *= f;
        ty *= f;
        tz *= f;

        double s = 2.0 / (1.0 + (tx * tx + ty * ty) + tz * tz);
        double sx = tx * s, sy = ty * s, sz = tz * s;

        double u0p = u0 + u1 * tz - u2 * ty;
        double u1p = u1 + u2 * tx - u0 * tz;
        double u2p = u2 + u0 * ty - u1 * tx;

        u0 += u1p * sz - u2p * sy;
        u1 += u2p * sx - u0p * sz;
        u2 += u0p * sy - u1p * sx;

        u0 += econst * Ex[i];
        u1 += econst * Ey[i];
        u2 += econst * Ez[i];

        ux[i] = u0;
        uy[i] = u1;
        uz[i] = u2;
    }
}

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type, const int64_t np_particles, const double q)
{
    const double econst = ((0.5 * q) * DT) / m;
    const int64_t mpt = momentum_push_type;
    const int64_t n = np_particles;
    if (mpt == 0)
        boris_full(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, n, econst);
    else if (mpt == 1)
        boris_first(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, n, econst);
    else
        boris_second(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, n, econst);
}
