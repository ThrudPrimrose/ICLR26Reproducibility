/* LDA exchange-correlation potential (Slater exchange + Perdew-Zunger correlation).
 *
 * Canonical C-ABI (abi_contract.md Sec. 4): ALL pointers sorted by name, then all
 * scalars/symbols by name, then the workspace pair:
 *     void lda_xc_potential_fp64(exc, rho, vxc, N, dvol, workspace, workspace_size)
 *
 * Matches the NumPy oracle element-for-element, including non-finite inputs:
 *   - n = max(rho,1e-12) must PROPAGATE NaN (np.maximum semantics, not fmax).
 *   - rs = (3/(4*pi*n))^(1/3) is formed EXACTLY as the oracle does,
 *     pow(3/(4*pi*n), 1/3), so 4*pi*n overflows to Inf at the same n (making
 *     rs 0, log(rs) -Inf and the correlation NaN) as in the reference.
 *   - n13 = C1/rs exploits rs*n13 == C1 == (3/(4*pi))^(1/3); it agrees with
 *     the oracle's pow(n,1/3) to a few ulp for finite n and both yield NaN
 *     (hence NaN vxc) where the oracle's rs is 0 or NaN.
 *   - PZ correlation is piecewise in rs; both branches are evaluated and
 *     selected with a mask blend (safe when the unselected side is non-finite).
 * pow/sqrt/log are vectorized through libmvec; the outer loop is threaded.
 */
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho,
                           double *restrict vxc, const int64_t N, const double dvol,
                           uint8_t *restrict workspace, const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const double AX    = 0.9847450218426965;
    const double GAMMA = -0.1423, B1 = 1.0529, B2 = 0.3334;   /* PZ, rs >= 1 */
    const double A = 0.0311, B = -0.0480, C = 0.0020, D = -0.0116; /* PZ, rs < 1 */
    const double Bm   = B - A / 3.0;
    const double C23  = (2.0 / 3.0) * C;
    const double D23  = (2.0 * D - C) / 3.0;
    const double B17  = (7.0 / 6.0) * B1;
    const double B24  = (4.0 / 3.0) * B2;
    const double C1   = pow(3.0 / (4.0 * M_PI), 1.0 / 3.0); /* (3/(4*pi))^(1/3) */

    const int64_t n3 = (int64_t)N * (int64_t)N * (int64_t)N;
    double sum = 0.0;

#pragma omp parallel for schedule(static) reduction(+:sum)
    for (int64_t i = 0; i < n3; i++) {
        double n       = (rho[i] < 1.0e-12) ? 1.0e-12 : rho[i]; /* NaN-propagating max */
        double rs      = pow(3.0 / (4.0 * M_PI * n), 1.0 / 3.0); /* oracle's rs */
        double n13     = C1 / rs;                                  /* n^(1/3) */
        double sqrt_rs = sqrt(rs);
        double ln_rs   = log(rs);

        double eps_x   = -0.75 * AX * n13;
        double v_x     = -AX * n13;
        double denom   = 1.0 + B1 * sqrt_rs + B2 * rs;
        double eps_ge1 = GAMMA / denom;
        double v_ge1   = eps_ge1 * (1.0 + B17 * sqrt_rs + B24 * rs) / denom;
        double eps_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
        double v_lt1   = A * ln_rs + Bm + C23 * rs * ln_rs + D23 * rs;

        double eps_c = rs < 1.0 ? eps_lt1 : eps_ge1; /* mask blend, NaN-safe */
        double v_c   = rs < 1.0 ? v_lt1   : v_ge1;

        vxc[i] = v_x + v_c;
        sum   += (eps_x + eps_c) * n;
    }
    exc[0] = dvol * sum;
}
