/* LDA XC potential/energy: Slater exchange + Perdew-Zunger correlation.
 * Optimized: single fused pass, one cbrt + one log per element, OMP-parallel
 * with a private reduction.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

static const double AX = 0.9847450218426965;  /* (3/pi)^(1/3)  */
static const double GAMMA = -0.1423;
static const double B1 = 1.0529;
static const double B2 = 0.3334;
static const double CA = 0.0311;
static const double CB = -0.0480;
static const double CC = 0.0020;
static const double CD = -0.0116;
static const double C0 = 0.6203504908143758;  /* (3/(4*pi))^(1/3) */

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N, const double dvol) {
    const int64_t ntot = (int64_t)N * N * N;
    const double KA = -0.75 * AX;
    const double K1 = (7.0 / 6.0) * B1;
    const double K2 = (4.0 / 3.0) * B2;
    const double K3 = (2.0 / 3.0) * CC;
    const double K4 = (2.0 * CD - CC) / 3.0;
    const double K5 = CB - CA / 3.0;

    double partial = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:partial)
    for (int64_t i = 0; i < ntot; ++i) {
        const double r = rho[i];
        const double n = (r != r) ? r : (1e-12 > r ? 1e-12 : r); /* np.maximum */
        const double n13 = cbrt(n);
        const double rs = C0 / n13;
        const double s = sqrt(rs);
        const double l = log(rs);
        const double denom = 1.0 + B1 * s + B2 * rs;
        const double e1 = GAMMA / denom;
        const double v1 = e1 * (1.0 + K1 * s + K2 * rs) / denom;
        const double e2 = CA * l + CB + CC * rs * l + CD * rs;
        const double v2 = CA * l + K5 + K3 * rs * l + K4 * rs;
        const int64_t hd = rs < 1.0;
        vxc[i] = -AX * n13 + (hd ? v2 : v1);
        partial += (KA * n13 + (hd ? e2 : e1)) * n;
    }
    exc[0] = dvol * partial;
}
