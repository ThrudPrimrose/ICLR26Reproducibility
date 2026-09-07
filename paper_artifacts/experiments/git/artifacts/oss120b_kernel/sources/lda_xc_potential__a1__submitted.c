#define _USE_MATH_DEFINES
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Constants from lda_xc_potential_numpy.py */
static const double _AX = 0.9847450218426965; // (3/pi)^(1/3), Slater-exchange coefficient
static const double _GAMMA = -0.1423, _B1 = 1.0529, _B2 = 0.3334; // Perdew-Zunger correlation, rs >= 1
static const double _A = 0.0311, _B = -0.0480, _C = 0.0020, _D = -0.0116; // Perdew-Zunger correlation, rs < 1

/*
 * LDA exchange-correlation potential and energy density.
 * Arguments order mirrors the Python kernel signature (dvol, rho, vxc, exc),
 * with the cube size N appended after the arrays (required for C-ABI).
 *
 *   dvol : cell volume (scalar)
 *   rho  : input density array (size N^3)
 *   vxc  : output potential array (size N^3)
 *   exc  : output array of length 1, receives the total xc energy (dvol * sum(...))
 *   N    : dimension of the cubic grid (rho has N*N*N elements)
 */
void lda_xc_potential_fp64(const double *restrict rho,
                            double *restrict vxc,
                            double *restrict exc,
                            int64_t N,
                            double dvol)
{
    const double eps = 1.0e-12;
    const int64_t total = N * N * N;
    double exc_sum = 0.0;

    #pragma omp parallel for reduction(+:exc_sum)
    for (int64_t i = 0; i < total; ++i) {
        if (isnan(rho[i])) { vxc[i] = NAN; exc_sum += NAN; continue; } double n = rho[i] > eps ? rho[i] : eps; // np.maximum(rho, eps)
        double n13 = cbrt(n);
        double rs = cbrt(3.0 / (4.0 * M_PI * n));
        double eps_x = -0.75 * _AX * n13;
        double v_x = -_AX * n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);
        double denom = 1.0 + _B1 * sqrt_rs + _B2 * rs;
        double eps_c_ge1 = _GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 * (1.0 + (7.0/6.0) * _B1 * sqrt_rs + (4.0/3.0) * _B2 * rs) / denom;
        double eps_c_lt1 = _A * ln_rs + _B + _C * rs * ln_rs + _D * rs;
        double v_c_lt1 = _A * ln_rs + (_B - _A/3.0) + (2.0/3.0) * _C * rs * ln_rs + (2.0*_D - _C) / 3.0 * rs;
        double high_density = rs < 1.0;
        double eps_c = high_density ? eps_c_lt1 : eps_c_ge1;
        double v_c = high_density ? v_c_lt1 : v_c_ge1;
        vxc[i] = v_x + v_c;
        exc_sum += (eps_x + eps_c) * n;
    }
    exc[0] = dvol * exc_sum;
}

