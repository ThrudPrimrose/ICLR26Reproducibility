#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const double INV_C2 = 1.1126500560536185e-17;
static const double DT = 1.0e-13;

enum { FULL = 0, FIRST_HALF = 1, SECOND_HALF = 2 };

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           double m, int64_t momentum_push_type, int64_t np_particles,
                           double q, uint8_t *restrict workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    const double econst = 0.5 * q * DT / m;
    const double inv_c2 = INV_C2;

    #pragma omp parallel for
    for (int64_t ip = 0; ip < np_particles; ++ip) {
        double x = ux[ip];
        double y = uy[ip];
        double z = uz[ip];

        if (momentum_push_type == FIRST_HALF || momentum_push_type == FULL) {
            x += econst * Ex[ip];
            y += econst * Ey[ip];
            z += econst * Ez[ip];
        }

        const double inv_gamma = 1.0 / sqrt(1.0 + (x * x + y * y + z * z) * inv_c2);

        double tx = econst * inv_gamma * Bx[ip];
        double ty = econst * inv_gamma * By[ip];
        double tz = econst * inv_gamma * Bz[ip];

        if (momentum_push_type == FIRST_HALF || momentum_push_type == SECOND_HALF) {
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
        const double ux_p = x + y * tz - z * ty;
        const double uy_p = y + z * tx - x * tz;
        const double uz_p = z + x * ty - y * tx;

        x += uy_p * sz - uz_p * sy;
        y += uz_p * sx - ux_p * sz;
        z += ux_p * sy - uy_p * sx;

        if (momentum_push_type == SECOND_HALF || momentum_push_type == FULL) {
            x += econst * Ex[ip];
            y += econst * Ey[ip];
            z += econst * Ez[ip];
        }

        ux[ip] = x;
        uy[ip] = y;
        uz[ip] = z;
    }
}
