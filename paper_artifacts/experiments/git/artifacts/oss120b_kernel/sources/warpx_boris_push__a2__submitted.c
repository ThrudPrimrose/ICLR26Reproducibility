// Optimized C implementation of the WarpX Boris momentum pusher.
// Updated to avoid potential compiler vectorization aliasing issues.
// The API matches the benchmark expected signature.

#include <math.h>
#ifdef _OPENMP
#define WARPX_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define WARPX_OMP_PARALLEL_FOR
#endif

static const int FULL = 0;
static const int FIRST_HALF = 1;
static const int SECOND_HALF = 2;

static const double C_LIGHT = 299792458.0;
static const double INV_C2 = 1.0 / (C_LIGHT * C_LIGHT);

void warpx_boris_push_fp64(const double *__restrict__ Bx,
                           const double *__restrict__ By,
                           const double *__restrict__ Bz,
                           const double *__restrict__ Ex,
                           const double *__restrict__ Ey,
                           const double *__restrict__ Ez,
                           double *__restrict__ ux,
                           double *__restrict__ uy,
                           double *__restrict__ uz,
                           double dt,
                           double m,
                           int momentum_push_type,
                           double q,
                           long np) {
    const double econst = 0.5 * q * dt / m;
    const double inv_c2 = INV_C2;

    WARPX_OMP_PARALLEL_FOR
    for (long ip = 0; ip < np; ++ip) {
        // Load current momenta (including first half electric push if applicable).
        double ux_val = ux[ip];
        double uy_val = uy[ip];
        double uz_val = uz[ip];

        if (momentum_push_type == FIRST_HALF || momentum_push_type == FULL) {
            ux_val += econst * Ex[ip];
            uy_val += econst * Ey[ip];
            uz_val += econst * Ez[ip];
        }

        // Compute gamma factor using the updated momentum.
        double inv_gamma = 1.0 / sqrt(1.0 + (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val) * inv_c2);

        // Magnetic rotation vector t.
        double tx = econst * inv_gamma * Bx[ip];
        double ty = econst * inv_gamma * By[ip];
        double tz = econst * inv_gamma * Bz[ip];

        // Rescale t for half pushes.
        if (momentum_push_type == FIRST_HALF || momentum_push_type == SECOND_HALF) {
            double tsq = tx*tx + ty*ty + tz*tz;
            double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
            tx *= factor;
            ty *= factor;
            tz *= factor;
        }

        // Compute rotation coefficients.
        double tsqi = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
        double sx = tx * tsqi;
        double sy = ty * tsqi;
        double sz = tz * tsqi;

        // Intermediate momentum (cross product with t).
        double ux_p = ux_val + uy_val * tz - uz_val * ty;
        double uy_p = uy_val + uz_val * tx - ux_val * tz;
        double uz_p = uz_val + ux_val * ty - uy_val * tx;

        // Apply magnetic rotation.
        double ux_rot = ux_val + uy_p * sz - uz_p * sy;
        double uy_rot = uy_val + uz_p * sx - ux_p * sz;
        double uz_rot = uz_val + ux_p * sy - uy_p * sx;

        // Second half electric push (if applicable).
        if (momentum_push_type == SECOND_HALF || momentum_push_type == FULL) {
            ux_rot += econst * Ex[ip];
            uy_rot += econst * Ey[ip];
            uz_rot += econst * Ez[ip];
        }

        // Store results back to memory.
        ux[ip] = ux_rot;
        uy[ip] = uy_rot;
        uz[ip] = uz_rot;
    }
}
