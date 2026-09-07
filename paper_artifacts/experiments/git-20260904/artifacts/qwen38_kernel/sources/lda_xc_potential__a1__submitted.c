#include <stdint.h>
#include <math.h>

/* LDA XC potential/energy: Slater exchange + Perdew-Zunger correlation.
   Fused single-pass map + reduce over the N^3 grid, multi-threaded. */
void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N, const double dvol) {
    const double AX = 0.9847450218426965;
    const double GAMMA = -0.1423, B1 = 1.0529, B2 = 0.3334;
    const double A = 0.0311, B = -0.0480, C = 0.0020, D = -0.0116;
    const double C1 = cbrt(3.0 / (4.0 * 3.14159265358979323846)); /* (3/(4*pi))^(1/3) */
    const double s76_B1 = (7.0 / 6.0) * B1;
    const double s43_B2 = (4.0 / 3.0) * B2;
    const double Bm_A3 = B - A / 3.0;
    const double s23_C = (2.0 / 3.0) * C;
    const double s2DmC3 = (2.0 * D - C) / 3.0;
    const long long total = (long long)N * (long long)N * (long long)N;
    double acc = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:acc)
    for (long long i = 0; i < total; ++i) {
        double r = rho[i];
        double n = r > 1e-12 ? r : 1e-12;
        double n13 = cbrt(n);
        double rs = C1 / n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);
        double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
        double eps_c_ge1 = GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 * (1.0 + s76_B1 * sqrt_rs + s43_B2 * rs) / denom;
        double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        double v_c_lt1 = A * ln_rs + Bm_A3 + s23_C * rs * ln_rs + s2DmC3 * rs;
        double eps_c = (rs < 1.0) ? eps_c_lt1 : eps_c_ge1;
        double v_c = (rs < 1.0) ? v_c_lt1 : v_c_ge1;
        double anx = AX * n13;
        vxc[i] = -anx + v_c;
        acc += (-0.75 * anx + eps_c) * n;
    }
    exc[0] = dvol * acc;
}
