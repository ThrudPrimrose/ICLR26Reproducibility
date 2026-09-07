#include <stdint.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lda_xc_potential_fp64(double *restrict exc,
                           const double *restrict rho,
                           double *restrict vxc,
                           const int64_t N,
                           const double dvol)
{
    const double AX = 0.9847450218426965;
    const double GAMMA = -0.1423;
    const double B1 = 1.0529;
    const double B2 = 0.3334;
    const double A = 0.0311;
    const double B = -0.0480;
    const double C = 0.0020;
    const double D = -0.0116;

    const double c13 = 1.0 / 3.0;
    const double inv_4pi_3 = 3.0 / (4.0 * M_PI);
    const double c0 = cbrt(inv_4pi_3);
    const double seven_sixths_B1 = (7.0 / 6.0) * B1;
    const double four_thirds_B2 = (4.0 / 3.0) * B2;
    const double two_thirds_C = (2.0 / 3.0) * C;
    const double twoD_minus_C_over3 = (2.0 * D - C) / 3.0;
    const double B_minus_A_over3 = B - A * c13;

    const int64_t total = N * N * N;
    double exc_acc = 0.0;

    #pragma omp parallel for simd reduction(+:exc_acc) schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        double n = rho[i];
        n = (n < 1.0e-12) ? 1.0e-12 : n;

        double n13 = cbrt(n);
        double eps_x = -0.75 * AX * n13;
        double v_x = -AX * n13;

        double rs = c0 / n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);

        double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
        double eps_c_ge1 = GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 *
                         (1.0 + seven_sixths_B1 * sqrt_rs + four_thirds_B2 * rs) /
                         denom;
        double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        double v_c_lt1 = A * ln_rs + B_minus_A_over3 +
                         two_thirds_C * rs * ln_rs +
                         twoD_minus_C_over3 * rs;

        double use_lt1 = (double)(rs < 1.0);
        double eps_c = use_lt1 * eps_c_lt1 + (1.0 - use_lt1) * eps_c_ge1;
        double v_c = use_lt1 * v_c_lt1 + (1.0 - use_lt1) * v_c_ge1;

        vxc[i] = v_x + v_c;
        exc_acc += (eps_x + eps_c) * n;
    }

    exc[0] = dvol * exc_acc;
}
