#define _USE_MATH_DEFINES
#include <stdint.h>
#include <math.h>
#include <omp.h>

constexpr double dt = 1e-13;

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type,
                           const int64_t np_particles, const double q) {
    const int64_t mpt = momentum_push_type;
    const double econst = ((0.5 * q) * dt) / m;
    const double inv_c2 = 1.1126500560536185e-17;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < np_particles; ++i) {
        double ux_i = ux[i];
        double uy_i = uy[i];
        double uz_i = uz[i];
        // First half electric push if needed
        if (mpt == 0 || mpt == 1) {
            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];
        }
        // Compute gamma factor
        double inv_gamma = 1.0 / sqrt(1.0 + (ux_i*ux_i + uy_i*uy_i + uz_i*uz_i) * inv_c2);
        // Magnetic rotation vector t
        double tx = econst * inv_gamma * Bx[i];
        double ty = econst * inv_gamma * By[i];
        double tz = econst * inv_gamma * Bz[i];
        // Rescale t if half push (first or second half)
        if (mpt == 1 || mpt == 2) {
            double tsq = tx*tx + ty*ty + tz*tz;
            double factor = (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;
            tx *= factor;
            ty *= factor;
            tz *= factor;
        }
        // Compute s components
        double tsqi = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
        double sx = tx * tsqi;
        double sy = ty * tsqi;
        double sz = tz * tsqi;
        // Intermediate momentum
        double ux_p = ux_i + uy_i * tz - uz_i * ty;
        double uy_p = uy_i + uz_i * tx - ux_i * tz;
        double uz_p = uz_i + ux_i * ty - uy_i * tx;
        // Update momentum
        ux_i += uy_p * sz - uz_p * sy;
        uy_i += uz_p * sx - ux_p * sz;
        uz_i += ux_p * sy - uy_p * sx;
        // Second half electric push if needed
        if (mpt == 0 || mpt == 2) {
            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];
        }
        ux[i] = ux_i;
        uy[i] = uy_i;
        uz[i] = uz_i;
    }
}