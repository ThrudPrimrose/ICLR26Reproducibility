/*
 * 8th-order (R=4) central-difference 3-D Laplacian on a periodic grid,
 * batched over k states, fused with the per-state kinetic energy.
 *
 *   lap[i,j,l,m] = inv_h2 * S(i,j,l,m)
 *   S = 3*C0*psi + sum_{axis, r=1..4} w_r * (psi[axis+/-r] + ...)   (periodic)
 *   ekin[m] = -0.5 * sum_{i,j,l} psi[i,j,l,m] * lap[i,j,l,m]
 *
 * C-ABI: laplacian_stencil_3d_fp64 (c-abi-v2)
 */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#define C0c   (-205.0 / 72.0)
#define W1c   (8.0 / 5.0)
#define W2c   (-1.0 / 5.0)
#define W3c   (8.0 / 315.0)
#define W4c   (-1.0 / 560.0)

#define KMAX 64
#define TMAX 1024

static double ekin_part[TMAX][KMAX];

void laplacian_stencil_3d_fp64(
    double *restrict ekin,
    double *restrict lap,
    const double *restrict psi,
    const int64_t N,
    const double inv_h2,
    const int64_t k,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    const double c03 = 3.0 * C0c;
    const int64_t N2 = N * N;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *ep = ekin_part[tid < TMAX ? tid : TMAX - 1];
        for (int64_t kk = 0; kk < k; kk++) ep[kk] = 0.0;

        #pragma omp for schedule(static) collapse(2)
        for (int64_t i = 0; i < N; i++) {
            int64_t im1 = i - 1; if (im1 < 0) im1 += N;
            int64_t im2 = i - 2; if (im2 < 0) im2 += N;
            int64_t im3 = i - 3; if (im3 < 0) im3 += N;
            int64_t im4 = i - 4; if (im4 < 0) im4 += N;
            int64_t ip1 = i + 1; if (ip1 >= N) ip1 -= N;
            int64_t ip2 = i + 2; if (ip2 >= N) ip2 -= N;
            int64_t ip3 = i + 3; if (ip3 >= N) ip3 -= N;
            int64_t ip4 = i + 4; if (ip4 >= N) ip4 -= N;

            const double *ci   = psi + i * N2 * k;
            const double *cim1 = psi + im1 * N2 * k;
            const double *cim2 = psi + im2 * N2 * k;
            const double *cim3 = psi + im3 * N2 * k;
            const double *cim4 = psi + im4 * N2 * k;
            const double *cip1 = psi + ip1 * N2 * k;
            const double *cip2 = psi + ip2 * N2 * k;
            const double *cip3 = psi + ip3 * N2 * k;
            const double *cip4 = psi + ip4 * N2 * k;

            for (int64_t j = 0; j < N; j++) {
                int64_t jm1 = j - 1; if (jm1 < 0) jm1 += N;
                int64_t jm2 = j - 2; if (jm2 < 0) jm2 += N;
                int64_t jm3 = j - 3; if (jm3 < 0) jm3 += N;
                int64_t jm4 = j - 4; if (jm4 < 0) jm4 += N;
                int64_t jp1 = j + 1; if (jp1 >= N) jp1 -= N;
                int64_t jp2 = j + 2; if (jp2 >= N) jp2 -= N;
                int64_t jp3 = j + 3; if (jp3 >= N) jp3 -= N;
                int64_t jp4 = j + 4; if (jp4 >= N) jp4 -= N;

                const double *pi    = ci + j * N * k;

                for (int64_t l = 0; l < N; l++) {
                    int64_t lm1 = l - 1; if (lm1 < 0) lm1 += N;
                    int64_t lm2 = l - 2; if (lm2 < 0) lm2 += N;
                    int64_t lm3 = l - 3; if (lm3 < 0) lm3 += N;
                    int64_t lm4 = l - 4; if (lm4 < 0) lm4 += N;
                    int64_t lp1 = l + 1; if (lp1 >= N) lp1 -= N;
                    int64_t lp2 = l + 2; if (lp2 >= N) lp2 -= N;
                    int64_t lp3 = l + 3; if (lp3 >= N) lp3 -= N;
                    int64_t lp4 = l + 4; if (lp4 >= N) lp4 -= N;

                    double *o = lap + (i * N + j) * N * k + l * k;

                    const double *c  = pi + l * k;
                    const double *z1 = pi + lm1 * k;
                    const double *z2 = pi + lm2 * k;
                    const double *z3 = pi + lm3 * k;
                    const double *z4 = pi + lm4 * k;
                    const double *z5 = pi + lp1 * k;
                    const double *z6 = pi + lp2 * k;
                    const double *z7 = pi + lp3 * k;
                    const double *z8 = pi + lp4 * k;

                    const double *pj_m1 = ci + jm1 * N * k + l * k;
                    const double *pj_m2 = ci + jm2 * N * k + l * k;
                    const double *pj_m3 = ci + jm3 * N * k + l * k;
                    const double *pj_m4 = ci + jm4 * N * k + l * k;
                    const double *pj_p1 = ci + jp1 * N * k + l * k;
                    const double *pj_p2 = ci + jp2 * N * k + l * k;
                    const double *pj_p3 = ci + jp3 * N * k + l * k;
                    const double *pj_p4 = ci + jp4 * N * k + l * k;

                    const double *xi_m1 = cim1 + j * N * k + l * k;
                    const double *xi_m2 = cim2 + j * N * k + l * k;
                    const double *xi_m3 = cim3 + j * N * k + l * k;
                    const double *xi_m4 = cim4 + j * N * k + l * k;
                    const double *xi_p1 = cip1 + j * N * k + l * k;
                    const double *xi_p2 = cip2 + j * N * k + l * k;
                    const double *xi_p3 = cip3 + j * N * k + l * k;
                    const double *xi_p4 = cip4 + j * N * k + l * k;

                    for (int64_t kk = 0; kk < k; kk++) {
                        double s = c03 * c[kk];
                        s += W1c * (xi_p1[kk] + xi_m1[kk] + pj_p1[kk] + pj_m1[kk] + z5[kk] + z1[kk]);
                        s += W2c * (xi_p2[kk] + xi_m2[kk] + pj_p2[kk] + pj_m2[kk] + z6[kk] + z2[kk]);
                        s += W3c * (xi_p3[kk] + xi_m3[kk] + pj_p3[kk] + pj_m3[kk] + z7[kk] + z3[kk]);
                        s += W4c * (xi_p4[kk] + xi_m4[kk] + pj_p4[kk] + pj_m4[kk] + z8[kk] + z4[kk]);
                        double lv = inv_h2 * s;
                        o[kk] = lv;
                        ep[kk] += c[kk] * lv;
                    }
                }
            }
        }

        #pragma omp for schedule(static)
        for (int64_t kk = 0; kk < k; kk++) {
            double sum = 0.0;
            for (int t = 0; t < omp_get_num_threads(); t++) sum += ekin_part[t][kk];
            ekin[kk] = -0.5 * sum;
        }
    }
}
