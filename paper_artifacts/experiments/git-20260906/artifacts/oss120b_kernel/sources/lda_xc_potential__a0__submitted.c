/*
 * LDA exchange-correlation (XC) potential and energy kernel.
 * Implements the reference algorithm from /shared/tasks/lda_xc_potential/lda_xc_potential_numpy.py.
 *
 * Arguments (C-ABI, double precision):
 *   const double *restrict rho   - input density array (size N^3)
 *   double *restrict vxc         - output XC potential array (size N^3)
 *   double *restrict exc         - output scalar energy (size 1)
 *   int64_t N                    - grid dimension (cube root of array size)
 *   double dvol                 - cell volume (constant per element)
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Constants for the LDA exchange and Perdew‑Zunger correlation */
static const double _AX    = 0.9847450218426965; /* (3/pi)^(1/3) */
static const double _GAMMA = -0.1423;
static const double _B1    = 1.0529;
static const double _B2    = 0.3334;
static const double _A     = 0.0311;
static const double _B     = -0.0480;
static const double _C     = 0.0020;
static const double _D     = -0.0116;

/* Main kernel – double precision version */
void lda_xc_potential_fp64(const double *restrict rho,
                           double *restrict vxc,
                           double *restrict exc,
                           int64_t N,
                           double dvol) {
    const size_t total = (size_t)N * (size_t)N * (size_t)N;
    double sum = 0.0;

    /* Parallel loop with reduction for the energy sum. */
#pragma omp parallel for schedule(static) reduction(+:sum)
    for (size_t i = 0; i < total; ++i) {
        if (isnan(rho[i])) {
    vxc[i] = NAN;
    sum += NAN;
    continue;
}
    double n = rho[i] > 1.0e-12 ? rho[i] : 1.0e-12;
        double rs = cbrt(3.0 / (4.0 * M_PI * n));
        double n13 = cbrt(n);
        double eps_x = -0.75 * _AX * n13;
        double v_x   = -_AX * n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs   = log(rs);
        double denom   = 1.0 + _B1 * sqrt_rs + _B2 * rs;
        double eps_c, v_c;
        if (rs < 1.0) {
            eps_c = _A * ln_rs + _B + _C * rs * ln_rs + _D * rs;
            v_c   = _A * ln_rs + (_B - _A / 3.0) + (2.0 / 3.0) * _C * rs * ln_rs + (2.0 * _D - _C) / 3.0 * rs;
        } else {
            eps_c = _GAMMA / denom;
            v_c   = eps_c * (1.0 + (7.0 / 6.0) * _B1 * sqrt_rs + (4.0 / 3.0) * _B2 * rs) / denom;
        }
        vxc[i] = v_x + v_c;
        sum += (eps_x + eps_c) * n;
    }

    /* Store the reduced energy. */
    exc[0] = dvol * sum;
}
