#include <stdint.h>
#include <math.h>
#include <omp.h>

static const double _AX = 0.9847450218426965;
static const double _GAMMA = -0.1423, _B1 = 1.0529, _B2 = 0.3334;
static const double _A = 0.0311, _B = -0.0480, _C = 0.0020, _D = -0.0116;

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N, const double dvol,
                           void *restrict ws, const int64_t wsz)
{
    const int64_t tot = N * N * N;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < tot; ++i) {
        double n = rho[i] >= 1e-12 ? rho[i] : 1e-12;
        double rs = pow(3.0 / ((4.0 * 3.141592653589793) * n), (1.0 / 3.0));
        double n13 = pow(n, (1.0 / 3.0));
        double eps_x = ((-0.75) * _AX) * n13;
        double v_x = (-_AX) * n13;
        double sqrt_rs = sqrt(rs);
        double ln_rs = log(rs);
        double denom = 1.0 + _B1 * sqrt_rs + _B2 * rs;
        double eps_c_ge1 = _GAMMA / denom;
        double v_c_ge1 = eps_c_ge1 * (1.0 + (7.0 / 6.0) * _B1 * sqrt_rs + (4.0 / 3.0) * _B2 * rs) / denom;
        double eps_c_lt1 = _A * ln_rs + _B + _C * rs * ln_rs + _D * rs;
        double v_c_lt1 = _A * ln_rs + (_B - _A / 3.0) + (2.0 / 3.0) * _C * rs * ln_rs + (2.0 * _D - _C) / 3.0 * rs;
        double eps_c, v_c;
        if (rs < 1.0) { eps_c = eps_c_lt1; v_c = v_c_lt1; }
        else          { eps_c = eps_c_ge1; v_c = v_c_ge1; }
        vxc[i] = v_x + v_c;
        sum += (eps_x + eps_c) * n;
    }
    exc[0] = dvol * sum;
}
