#include <math.h>
#include <stdint.h>
#include <omp.h>

const double dt = 1e-13;
const double inv_c2 = 1.1126500560536185e-17;

static inline void boris_push_one(const double Bx_i,
                                   const double By_i,
                                   const double Bz_i,
                                   const double Ex_i,
                                   const double Ey_i,
                                   const double Ez_i,
                                   double *restrict ux_i,
                                   double *restrict uy_i,
                                   double *restrict uz_i,
                                   const double econst,
                                   const int do_first_push,
                                   const int do_factor,
                                   const int do_second_push)
{
    double ux = *ux_i;
    double uy = *uy_i;
    double uz = *uz_i;

    if (do_first_push) {
        ux += econst * Ex_i;
        uy += econst * Ey_i;
        uz += econst * Ez_i;
    }

    const double inv_gamma = 1.0 / sqrt(1.0 +
                                        (((ux * ux) + (uy * uy)) + (uz * uz)) *
                                        inv_c2);

    const double alpha = econst * inv_gamma;
    double tx = alpha * Bx_i;
    double ty = alpha * By_i;
    double tz = alpha * Bz_i;

    if (do_factor) {
        const double tsq = ((tx * tx) + (ty * ty)) + (tz * tz);
        if (tsq > 0.0) {
            const double factor = (sqrt(1.0 + tsq) - 1.0) / tsq;
            tx *= factor;
            ty *= factor;
            tz *= factor;
        }
    }

    const double tsqi = 2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz));
    const double sx = tx * tsqi;
    const double sy = ty * tsqi;
    const double sz = tz * tsqi;

    const double ux_p = (ux + (uy * tz)) - (uz * ty);
    const double uy_p = (uy + (uz * tx)) - (ux * tz);
    const double uz_p = (uz + (ux * ty)) - (uy * tx);

    ux += ((uy_p * sz) - (uz_p * sy));
    uy += ((uz_p * sx) - (ux_p * sz));
    uz += ((ux_p * sy) - (uy_p * sx));

    if (do_second_push) {
        ux += econst * Ex_i;
        uy += econst * Ey_i;
        uz += econst * Ez_i;
    }

    *ux_i = ux;
    *uy_i = uy;
    *uz_i = uz;
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
                           const int64_t np_particles,
                           const double q)
{
    const double econst = (0.5 * q) * dt / m;

    if (momentum_push_type == 0) {
        #pragma omp parallel for simd if(np_particles > 3000)
        for (int64_t i = 0; i < np_particles; ++i) {
            boris_push_one(Bx[i], By[i], Bz[i], Ex[i], Ey[i], Ez[i],
                           &ux[i], &uy[i], &uz[i], econst, 1, 0, 1);
        }
    } else if (momentum_push_type == 1) {
        #pragma omp parallel for simd if(np_particles > 3000)
        for (int64_t i = 0; i < np_particles; ++i) {
            boris_push_one(Bx[i], By[i], Bz[i], Ex[i], Ey[i], Ez[i],
                           &ux[i], &uy[i], &uz[i], econst, 1, 1, 0);
        }
    } else {
        #pragma omp parallel for simd if(np_particles > 3000)
        for (int64_t i = 0; i < np_particles; ++i) {
            boris_push_one(Bx[i], By[i], Bz[i], Ex[i], Ey[i], Ez[i],
                           &ux[i], &uy[i], &uz[i], econst, 0, 1, 1);
        }
    }
}
