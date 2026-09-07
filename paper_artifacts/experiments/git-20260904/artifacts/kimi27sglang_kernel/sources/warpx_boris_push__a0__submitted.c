#include <stdint.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const double INV_C2 = 1.1126500560536185e-17;
static const double DT = 1e-13;

enum { FULL = 0, FIRST_HALF = 1, SECOND_HALF = 2 };

static inline void push_particle(double x, double y, double z,
                                 double ex, double ey, double ez,
                                 double bx, double by, double bz,
                                 double econst, double inv_c2, int mpt,
                                 double *restrict ox, double *restrict oy, double *restrict oz)
{
    if (mpt == FIRST_HALF || mpt == FULL) {
        x += econst * ex;
        y += econst * ey;
        z += econst * ez;
    }

    const double inv_gamma = 1.0 / sqrt(1.0 + (x * x + y * y + z * z) * inv_c2);

    double tx = econst * inv_gamma * bx;
    double ty = econst * inv_gamma * by;
    double tz = econst * inv_gamma * bz;

    if (mpt == FIRST_HALF || mpt == SECOND_HALF) {
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

    if (mpt == SECOND_HALF || mpt == FULL) {
        x += econst * ex;
        y += econst * ey;
        z += econst * ez;
    }

    *ox = x;
    *oy = y;
    *oz = z;
}

void warpx_boris_push_fp64(const double *restrict Bx,
                           const double *restrict By,
                           const double *restrict Bz,
                           const double *restrict Ex,
                           const double *restrict Ey,
                           const double *restrict Ez,
                           double *restrict ux,
                           double *restrict uy,
                           double *restrict uz,
                           const double m,
                           const int64_t momentum_push_type,
                           const int64_t np,
                           const double q,
                           uint8_t *restrict workspace,
                           int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const double econst = 0.5 * q * DT / m;
    const double inv_c2 = INV_C2;
    const int mpt = (int)momentum_push_type;

    Bx = (const double *)__builtin_assume_aligned(Bx, 64);
    By = (const double *)__builtin_assume_aligned(By, 64);
    Bz = (const double *)__builtin_assume_aligned(Bz, 64);
    Ex = (const double *)__builtin_assume_aligned(Ex, 64);
    Ey = (const double *)__builtin_assume_aligned(Ey, 64);
    Ez = (const double *)__builtin_assume_aligned(Ez, 64);
    ux = (double *)__builtin_assume_aligned(ux, 64);
    uy = (double *)__builtin_assume_aligned(uy, 64);
    uz = (double *)__builtin_assume_aligned(uz, 64);

    switch (mpt) {
    case FULL:
        #pragma omp parallel for simd aligned(Bx,By,Bz,Ex,Ey,Ez,ux,uy,uz:64) schedule(static) if(np > 16384)
        for (int64_t ip = 0; ip < np; ++ip) {
            push_particle(ux[ip], uy[ip], uz[ip],
                          Ex[ip], Ey[ip], Ez[ip],
                          Bx[ip], By[ip], Bz[ip],
                          econst, inv_c2, FULL,
                          &ux[ip], &uy[ip], &uz[ip]);
        }
        break;
    case FIRST_HALF:
        #pragma omp parallel for simd aligned(Bx,By,Bz,Ex,Ey,Ez,ux,uy,uz:64) schedule(static) if(np > 16384)
        for (int64_t ip = 0; ip < np; ++ip) {
            push_particle(ux[ip], uy[ip], uz[ip],
                          Ex[ip], Ey[ip], Ez[ip],
                          Bx[ip], By[ip], Bz[ip],
                          econst, inv_c2, FIRST_HALF,
                          &ux[ip], &uy[ip], &uz[ip]);
        }
        break;
    case SECOND_HALF:
        #pragma omp parallel for simd aligned(Bx,By,Bz,Ex,Ey,Ez,ux,uy,uz:64) schedule(static) if(np > 16384)
        for (int64_t ip = 0; ip < np; ++ip) {
            push_particle(ux[ip], uy[ip], uz[ip],
                          Ex[ip], Ey[ip], Ez[ip],
                          Bx[ip], By[ip], Bz[ip],
                          econst, inv_c2, SECOND_HALF,
                          &ux[ip], &uy[ip], &uz[ip]);
        }
        break;
    }
}
