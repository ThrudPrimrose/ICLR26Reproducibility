// Hand-optimized LDA XC potential: single fused pass, no temporaries,
// one cbrt + one log + one sqrt + one division per grid point,
// OpenMP-parallel with a long-double reduction for the exchange-correlation energy.
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N, const double dvol)
{
    const int64_t nn = N * N * N;
    const double K = 0.6203504908994;          /* cbrt(3/(4*pi)):  rs = K/n13   */
    const double LOGK = -0.4774706527670604;   /* log(K): ln_rs = LOGK - log(n13) */
    const double AX = 0.9847450218426965;
    const double B1_76 = (7.0 / 6.0) * 1.0529;
    const double B2_43 = (4.0 / 3.0) * 0.3334;
    const double Bm = -0.0480 - (0.0311 / 3.0);
    const double K23 = (2.0 / 3.0) * 0.002;
    const double Kv = ((2.0 * -0.0116) - 0.002) / 3.0;

    long double acc = 0.0L;
    #pragma omp parallel for schedule(static) reduction(+:acc)
    for (int64_t i = 0; i < nn; ++i) {
        const double n = rho[i] > 1e-12 ? rho[i] : 1e-12;
        const double n13 = cbrt(n);
        const double rs = K / n13;
        const double ln_rs = LOGK - log(n13);
        const double sqrt_rs = sqrt(rs);
        const double denom = 1.0 + 1.0529 * sqrt_rs + 0.3334 * rs;
        const double eps_c_ge1 = -0.1423 / denom;
        const double v_c_ge1 = (eps_c_ge1 * (1.0 + B1_76 * sqrt_rs + B2_43 * rs)) / denom;
        const double eps_c_lt1 = 0.0311 * ln_rs + -0.048 + (0.002 * rs) * ln_rs + -0.0116 * rs;
        const double v_c_lt1 = 0.0311 * ln_rs + Bm + (K23 * rs) * ln_rs + Kv * rs;
        const double v_c = rs < 1.0 ? v_c_lt1 : v_c_ge1;
        const double eps_c = rs < 1.0 ? eps_c_lt1 : eps_c_ge1;
        vxc[i] = -AX * n13 + v_c;
        acc += (-0.75 * AX * n13 + eps_c) * n;
    }
    exc[0] = dvol * (double)acc;
}
