#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double npb_fmax(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return (a > b) ? a : b;
}

void lda_xc_potential_fp64(double *restrict exc,
                           const double *restrict rho,
                           double *restrict vxc,
                           const int64_t N,
                           const double dvol) {
    const double AX = 0.9847450218426965;
    const double GAMMA = -0.1423;
    const double B1 = 1.0529;
    const double B2 = 0.3334;
    const double A = 0.0311;
    const double B = -0.0480;
    const double C = 0.0020;
    const double D = -0.0116;

    const double fourpi = 4.0 * M_PI;
    const double onethird = 1.0 / 3.0;
    const double seven_sixths_b1 = (7.0 / 6.0) * B1;
    const double four_thirds_b2 = (4.0 / 3.0) * B2;
    const double two_thirds_c = (2.0 / 3.0) * C;
    const double v_c_lt1_b = B - A * onethird;
    const double v_c_lt1_d = (2.0 * D - C) * onethird;

    const size_t total = (size_t)N * (size_t)N * (size_t)N;
    const double rho_floor = 1.0e-12;

    double exc_sum = 0.0;

    #pragma omp parallel for reduction(+:exc_sum) schedule(static)
    for (size_t i = 0; i < total; ++i) {
        const double r = rho[i];
        const double n = npb_fmax(r, rho_floor);

        const double n13 = cbrt(n);
        const double rs  = cbrt(3.0 / (fourpi * n));

        const double eps_x = -0.75 * AX * n13;
        const double v_x   = -AX * n13;

        const double sqrt_rs = sqrt(rs);
        const double ln_rs   = log(rs);

        const double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
        const double eps_c_ge1 = GAMMA / denom;
        const double v_c_ge1 = eps_c_ge1 * (1.0 + seven_sixths_b1 * sqrt_rs + four_thirds_b2 * rs) / denom;

        const double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        const double v_c_lt1 = A * ln_rs + v_c_lt1_b + two_thirds_c * rs * ln_rs + v_c_lt1_d * rs;

        const bool high_density = rs < 1.0;
        const double eps_c = high_density ? eps_c_lt1 : eps_c_ge1;
        const double v_c   = high_density ? v_c_lt1   : v_c_ge1;

        vxc[i] = v_x + v_c;
        exc_sum += (eps_x + eps_c) * n;
    }

    exc[0] = dvol * exc_sum;
}
