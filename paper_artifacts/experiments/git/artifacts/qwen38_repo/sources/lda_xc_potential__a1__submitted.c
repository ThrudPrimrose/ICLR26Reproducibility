/* LDA XC potential/energy: Slater exchange + Perdew-Zunger correlation.
 *
 * Optimized: single fused pass over the grid (no temporaries), vectorizable
 * libm (via libmvec), OpenMP parallelism with per-thread partial sums.
 * Per-element math follows the reference's operation order (bit-level the
 * same libm calls, same associativity).
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* reference constants */
#define C_PI 3.141592653589793
#define C_AX 0.9847450218426965   /* (3/pi)^(1/3), Slater coefficient       */
#define C_GAMMA (-0.1423)
#define C_B1 1.0529
#define C_B2 0.3334
#define C_A 0.0311
#define C_B (-0.0480)
#define C_C 0.0020
#define C_D (-0.0116)

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N,
                           const double dvol)
{
    const int64_t total = N * N * N;
    const double four_pi = 4.0 * C_PI;
    const double one_third = 1.0 / 3.0;

    double sum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int64_t i = 0; i < total; ++i) {
        const double r = rho[i];
        const double n = r >= 1.0e-12 ? r : 1.0e-12;
        const double rs  = pow(3.0 / (four_pi * n), one_third);
        const double n13 = pow(n, one_third);
        const double sx  = sqrt(rs);
        const double lx  = log(rs);
        const double den = (1.0 + C_B1 * sx) + C_B2 * rs;
        const double eps_ge = C_GAMMA / den;
        const double v_ge   = eps_ge * ((1.0 + (7.0 / 6.0) * C_B1) * sx) + 0.0;
        const double eps_lt = (C_A * lx + C_B) + (C_C * rs) * lx + C_D * rs;
        const double v_lt   = (C_A * lx + (C_B - C_A / 3.0)) + ((2.0 / 3.0) * C_C * rs) * lx + ((2.0 * C_D - C_C) / 3.0) * rs;
        const int high = rs < 1.0;
        const double eps_x = (-0.75) * C_AX * n13;
        const double v_x   = -C_AX * n13;
        const double eps_c = high ? eps_lt : eps_ge;
        const double v_c   = high ? v_lt : (eps_ge * ((1.0 + (7.0 / 6.0) * C_B1) * sx + (4.0 / 3.0) * C_B2 * rs) / den);
        vxc[i] = v_x + v_c;
        sum += (eps_x + eps_c) * n;
    }
    exc[0] = dvol * sum;
}
