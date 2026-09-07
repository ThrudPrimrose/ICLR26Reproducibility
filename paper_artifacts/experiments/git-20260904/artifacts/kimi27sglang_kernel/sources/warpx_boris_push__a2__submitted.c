#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const double dt = 1e-13;
static const double inv_c2 = 1.1126500560536185e-17;

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type, const int64_t np_particles, const double q) {
    const double econst = 0.5 * q * dt / m;
    const int64_t mpt = momentum_push_type;
    const int64_t n = np_particles;

    const double *b_x = __builtin_assume_aligned(Bx, 64);
    const double *b_y = __builtin_assume_aligned(By, 64);
    const double *b_z = __builtin_assume_aligned(Bz, 64);
    const double *e_x = __builtin_assume_aligned(Ex, 64);
    const double *e_y = __builtin_assume_aligned(Ey, 64);
    const double *e_z = __builtin_assume_aligned(Ez, 64);
    double *u_x = __builtin_assume_aligned(ux, 64);
    double *u_y = __builtin_assume_aligned(uy, 64);
    double *u_z = __builtin_assume_aligned(uz, 64);

#pragma omp parallel for simd schedule(static) aligned(Bx,By,Bz,Ex,Ey,Ez,ux,uy,uz:64)
    for (int64_t ip = 0; ip < n; ++ip) {
        double uux = u_x[ip];
        double uuy = u_y[ip];
        double uuz = u_z[ip];

        if (mpt == 1 || mpt == 0) {
            uux += econst * e_x[ip];
            uuy += econst * e_y[ip];
            uuz += econst * e_z[ip];
        }

        const double inv_gamma = 1.0 / sqrt(1.0 + (uux * uux + uuy * uuy + uuz * uuz) * inv_c2);

        double tx = econst * inv_gamma * b_x[ip];
        double ty = econst * inv_gamma * b_y[ip];
        double tz = econst * inv_gamma * b_z[ip];

        if (mpt == 1 || mpt == 2) {
            const double tsq = tx * tx + ty * ty + tz * tz;
            const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
            tx *= factor;
            ty *= factor;
            tz *= factor;
        }

        const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        const double sx = tx * tsqi;
        const double sy = ty * tsqi;
        const double sz = tz * tsqi;

        const double ux_p = uux + uuy * tz - uuz * ty;
        const double uy_p = uuy + uuz * tx - uux * tz;
        const double uz_p = uuz + uux * ty - uuy * tx;

        uux += uy_p * sz - uz_p * sy;
        uuy += uz_p * sx - ux_p * sz;
        uuz += ux_p * sy - uy_p * sx;

        if (mpt == 2 || mpt == 0) {
            uux += econst * e_x[ip];
            uuy += econst * e_y[ip];
            uuz += econst * e_z[ip];
        }

        u_x[ip] = uux;
        u_y[ip] = uuy;
        u_z[ip] = uuz;
    }
}
