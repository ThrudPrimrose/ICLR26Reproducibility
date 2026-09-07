#include <stdint.h>
#include <math.h>

constexpr double dt = 1e-13;

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type, const int64_t np_particles, const double q) {
    const double econst = (0.5 * q * dt) / m;
    const double inv_c2 = 1.1126500560536185e-17;
    const int64_t mpt = momentum_push_type;
    const int64_t n = np_particles;

    if (mpt == 0) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            double ux_i = ux[i];
            double uy_i = uy[i];
            double uz_i = uz[i];

            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];

            double gamma = sqrt(1.0 + (ux_i * ux_i + uy_i * uy_i + uz_i * uz_i) * inv_c2);
            double inv_gamma = 1.0 / gamma;

            double tx = econst * inv_gamma * Bx[i];
            double ty = econst * inv_gamma * By[i];
            double tz = econst * inv_gamma * Bz[i];

            double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
            double sx = tx * tsqi;
            double sy = ty * tsqi;
            double sz = tz * tsqi;

            double ux_p = ux_i + uy_i * tz - uz_i * ty;
            double uy_p = uy_i + uz_i * tx - ux_i * tz;
            double uz_p = uz_i + ux_i * ty - uy_i * tx;

            ux_i += uy_p * sz - uz_p * sy;
            uy_i += uz_p * sx - ux_p * sz;
            uz_i += ux_p * sy - uy_p * sx;

            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];

            ux[i] = ux_i;
            uy[i] = uy_i;
            uz[i] = uz_i;
        }
    } else if (mpt == 1) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            double ux_i = ux[i];
            double uy_i = uy[i];
            double uz_i = uz[i];

            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];

            double gamma = sqrt(1.0 + (ux_i * ux_i + uy_i * uy_i + uz_i * uz_i) * inv_c2);
            double inv_gamma = 1.0 / gamma;

            double tx = econst * inv_gamma * Bx[i];
            double ty = econst * inv_gamma * By[i];
            double tz = econst * inv_gamma * Bz[i];

            double tsq = tx * tx + ty * ty + tz * tz;
            if (tsq > 0.0) {
                double factor = (sqrt(1.0 + tsq) - 1.0) / tsq;
                tx *= factor;
                ty *= factor;
                tz *= factor;
            }

            double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
            double sx = tx * tsqi;
            double sy = ty * tsqi;
            double sz = tz * tsqi;

            double ux_p = ux_i + uy_i * tz - uz_i * ty;
            double uy_p = uy_i + uz_i * tx - ux_i * tz;
            double uz_p = uz_i + ux_i * ty - uy_i * tx;

            ux_i += uy_p * sz - uz_p * sy;
            uy_i += uz_p * sx - ux_p * sz;
            uz_i += ux_p * sy - uy_p * sx;

            ux[i] = ux_i;
            uy[i] = uy_i;
            uz[i] = uz_i;
        }
    } else if (mpt == 2) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            double ux_i = ux[i];
            double uy_i = uy[i];
            double uz_i = uz[i];

            double gamma = sqrt(1.0 + (ux_i * ux_i + uy_i * uy_i + uz_i * uz_i) * inv_c2);
            double inv_gamma = 1.0 / gamma;

            double tx = econst * inv_gamma * Bx[i];
            double ty = econst * inv_gamma * By[i];
            double tz = econst * inv_gamma * Bz[i];

            double tsq = tx * tx + ty * ty + tz * tz;
            if (tsq > 0.0) {
                double factor = (sqrt(1.0 + tsq) - 1.0) / tsq;
                tx *= factor;
                ty *= factor;
                tz *= factor;
            }

            double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
            double sx = tx * tsqi;
            double sy = ty * tsqi;
            double sz = tz * tsqi;

            double ux_p = ux_i + uy_i * tz - uz_i * ty;
            double uy_p = uy_i + uz_i * tx - ux_i * tz;
            double uz_p = uz_i + ux_i * ty - uy_i * tx;

            ux_i += uy_p * sz - uz_p * sy;
            uy_i += uz_p * sx - ux_p * sz;
            uz_i += ux_p * sy - uy_p * sx;

            ux_i += econst * Ex[i];
            uy_i += econst * Ey[i];
            uz_i += econst * Ez[i];

            ux[i] = ux_i;
            uy[i] = uy_i;
            uz[i] = uz_i;
        }
    }
}
