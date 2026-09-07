#include <math.h>
#include <stdint.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AX 0.9847450218426965
#define GAMMA -0.1423
#define B1 1.0529
#define B2 0.3334
#define A 0.0311
#define B -0.0480
#define C 0.0020
#define D -0.0116

void lda_xc_potential_fp64(double *vxc, double *exc, double *rho, int64_t N,
                           double dvol, uint8_t *workspace, int64_t workspace_bytes)
{
    const double one_third = 1.0 / 3.0;
    const int64_t total = N * N * N;
    double exc_acc = 0.0;

    #pragma omp parallel for reduction(+:exc_acc) schedule(static)
    for (int64_t i = 0; i < total; i++) {
        double n = rho[i];
        if (n < 1.0e-12) n = 1.0e-12;

        double rs = pow(3.0 / (4.0 * M_PI * n), one_third);
        double n13 = pow(n, one_third);

        double eps_x = -0.75 * AX * n13;
        double v_x = -AX * n13;

        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);
        double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
        double eps_c_ge1 = GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 * (1.0 + (7.0 / 6.0) * B1 * sqrt_rs + (4.0 / 3.0) * B2 * rs) / denom;

        double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        double v_c_lt1 = A * ln_rs + (B - A * one_third) + (2.0 / 3.0) * C * rs * ln_rs + (2.0 * D - C) / 3.0 * rs;

        int high_density = rs < 1.0;
        double eps_c = high_density ? eps_c_lt1 : eps_c_ge1;
        double v_c = high_density ? v_c_lt1 : v_c_ge1;

        vxc[i] = v_x + v_c;
        exc_acc += (eps_x + eps_c) * n;
    }

    exc[0] = dvol * exc_acc;
}
