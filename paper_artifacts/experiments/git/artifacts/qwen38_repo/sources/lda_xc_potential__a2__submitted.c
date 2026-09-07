#define _USE_MATH_DEFINES
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* LDA exchange-correlation potential (Slater exchange + Perdew-Zunger LDA c).
 * Fused single-pass elementwise computation with one reduction for exc.
 * Elementwise expressions mirror the NumPy reference bit-for-bit (same libm
 * pow/log on the same intermediate values); the only permitted reassociation
 * is the per-thread reduction order of the final sum, which NumPy's own
 * np.sum does not define bitwise either. */
void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho, double *restrict vxc,
                           const int64_t N, const double dvol) {
    const double AX = 0.9847450218426965;
    const double GAMMA = -0.1423, B1 = 1.0529, B2 = 0.3334;
    const double A = 0.0311, B = -0.0480, C = 0.0020, D = -0.0116;
    const double FOUR_PI = 4.0 * M_PI; /* exact: power-of-two scaling of M_PI */
    const double C1 = 7.0 / 6.0, C2 = 4.0 / 3.0;
    const double V1 = C1 * B1, V2 = C2 * B2;
    const double WL0 = B - A / 3.0, WL2 = (2.0 / 3.0) * C, WL3 = (2.0 * D - C) / 3.0;

    const int64_t total_n = N * N * N;
    double sum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int64_t i = 0; i < total_n; ++i) {
        double r = rho[i];
        /* np.maximum(rho, 1e-12) -- propagates NaN like numpy */
        double n = (r != r) ? r : (1e-12 < r ? r : 1e-12);
        double rs = pow(3.0 / (FOUR_PI * n), 1.0 / 3.0);
        double n13 = pow(n, 1.0 / 3.0);
        double eps_x = -0.75 * AX * n13;
        double v_x = -AX * n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);
        double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
        double eps_c_ge1 = GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 * (1.0 + V1 * sqrt_rs + V2 * rs) / denom;
        double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        double v_c_lt1 = A * ln_rs + WL0 + WL2 * rs * ln_rs + WL3 * rs;
        double eps_c, v_c;
        if (rs < 1.0) {
            eps_c = eps_c_lt1;
            v_c = v_c_lt1;
        } else {
            eps_c = eps_c_ge1;
            v_c = v_c_ge1;
        }
        vxc[i] = v_x + v_c;
        sum += (eps_x + eps_c) * n;
    }
    exc[0] = dvol * sum;
}
