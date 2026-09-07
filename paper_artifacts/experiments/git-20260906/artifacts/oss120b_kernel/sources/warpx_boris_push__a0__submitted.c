// Optimized C implementation of the WarpX Boris momentum pusher.
// The kernel updates particle momenta in place.
// Signature matches the HPCAgent‑Bench auto‑generated binding.

#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#define WARPX_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define WARPX_OMP_PARALLEL_FOR
#endif

// Momentum push type enum values (matching the reference).
static const int64_t FULL = 0;
static const int64_t FIRST_HALF = 1;
static const int64_t SECOND_HALF = 2;

// Physical constants.
static const double C_LIGHT = 299792458.0;
static const double INV_C2 = 1.0 / (C_LIGHT * C_LIGHT);

// Fixed timestep for the benchmark (from manifest).
static const double DT = 1e-13;

// C‑ABI function signature matching the binding JSON:
// Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, m, momentum_push_type, np_particles, q
void warpx_boris_push_fp64(const double *__restrict__ Bx,
                           const double *__restrict__ By,
                           const double *__restrict__ Bz,
                           const double *__restrict__ Ex,
                           const double *__restrict__ Ey,
                           const double *__restrict__ Ez,
                           double *__restrict__ ux,
                           double *__restrict__ uy,
                           double *__restrict__ uz,
                           const double m,
                           const int64_t momentum_push_type,
                           const int64_t np_particles,
                           const double q) {
    const double econst = 0.5 * q * DT / m;
    const double inv_c2 = INV_C2;

    WARPX_OMP_PARALLEL_FOR
    for (int64_t ip = 0; ip < np_particles; ++ip) {
        // First half electric field push (if applicable).
        if (momentum_push_type == FIRST_HALF || momentum_push_type == FULL) {
            ux[ip] += econst * Ex[ip];
            uy[ip] += econst * Ey[ip];
            uz[ip] += econst * Ez[ip];
        }

        // Compute gamma factor.
        double inv_gamma = 1.0 / sqrt(1.0 + (ux[ip]*ux[ip] + uy[ip]*uy[ip] + uz[ip]*uz[ip]) * inv_c2);

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

        // Intermediate momentum.
        double ux_p = ux[ip] + uy[ip] * tz - uz[ip] * ty;
        double uy_p = uy[ip] + uz[ip] * tx - ux[ip] * tz;
        double uz_p = uz[ip] + ux[ip] * ty - uy[ip] * tx;

        // Apply magnetic rotation.
        ux[ip] += uy_p * sz - uz_p * sy;
        uy[ip] += uz_p * sx - ux_p * sz;
        uz[ip] += ux_p * sy - uy_p * sx;

        // Second half electric field push (if applicable).
        if (momentum_push_type == SECOND_HALF || momentum_push_type == FULL) {
            ux[ip] += econst * Ex[ip];
            uy[ip] += econst * Ey[ip];
            uz[ip] += econst * Ez[ip];
        }
    }
}

