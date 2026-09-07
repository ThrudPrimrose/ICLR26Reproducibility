// Optimized implementation of the WarpX Boris momentum pusher.
// Implements the same API as the reference reference.cpp.
// Expects double-precision data and all pointers are non-overlapping.
// The function is thread-parallelized with OpenMP and vector-friendly.
// Author: autonomous optimizer.

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

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

// Inline helper for a single particle update.
static inline void update_momentum_boris(double *restrict ux, double *restrict uy, double *restrict uz,
                                          const double Ex, const double Ey, const double Ez,
                                          const double Bx, const double By, const double Bz,
                                          const double q, const double m, const double dt,
                                          const int momentum_push_type) {
    const double econst = 0.5 * q * dt / m;

    // First half electric push.
    if (momentum_push_type == FIRST_HALF || momentum_push_type == FULL) {
        *ux += econst * Ex;
        *uy += econst * Ey;
        *uz += econst * Ez;
    }

    // Gamma factor.
    const double inv_gamma = 1.0 / sqrt(1.0 + ((*ux) * (*ux) + (*uy) * (*uy) + (*uz) * (*uz)) * INV_C2);

    // Magnetic rotation temporary vector t.
    double tx = econst * inv_gamma * Bx;
    double ty = econst * inv_gamma * By;
    double tz = econst * inv_gamma * Bz;

    if (momentum_push_type == FIRST_HALF || momentum_push_type == SECOND_HALF) {
        const double tsq = tx * tx + ty * ty + tz * tz;
        double factor = 0.5; // default for zero field.
        if (tsq > 0.0) {
            factor = (sqrt(1.0 + tsq) - 1.0) / tsq;
        }
        tx *= factor;
        ty *= factor;
        tz *= factor;
    }

    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi;
    const double sy = ty * tsqi;
    const double sz = tz * tsqi;
    const double ux_p = *ux + (*uy) * tz - (*uz) * ty;
    const double uy_p = *uy + (*uz) * tx - (*ux) * tz;
    const double uz_p = *uz + (*ux) * ty - (*uy) * tx;

    // Update momentum.
    *ux += uy_p * sz - uz_p * sy;
    *uy += uz_p * sx - ux_p * sz;
    *uz += ux_p * sy - uy_p * sx;

    // Second half electric push.
    if (momentum_push_type == SECOND_HALF || momentum_push_type == FULL) {
        *ux += econst * Ex;
        *uy += econst * Ey;
        *uz += econst * Ez;
    }
}

// C-ABI entry point. The name matches the benchmark harness expectation.
void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By,
                                 const double *restrict Bz, const double *restrict Ex,
                                 const double *restrict Ey, const double *restrict Ez,
                                 double *restrict ux, double *restrict uy, double *restrict uz,
                                 double m, double q, int64_t momentum_push_type,
                                 double dt, const uint8_t *workspace, int64_t workspace_bytes) {
    // Debug: write parameters to file (optional)
    FILE *debug_f = fopen("/shared/agent-13/debug.txt", "a");
    if (debug_f) {
        fprintf(debug_f, "m=%g q=%g momentum_push_type=%d dt=%g np=%ld\n", m, q, momentum_push_type, dt, np);
        fclose(debug_f);
    }
    // Parallel loop over particles.
    WARPX_OMP_PARALLEL_FOR
    for (int64_t i = 0; i < np; ++i) {
        update_momentum_boris(&ux[i], &uy[i], &uz[i],
                              Ex[i], Ey[i], Ez[i],
                              Bx[i], By[i], Bz[i],
                              q, m, dt, momentum_push_type);
    }
}
