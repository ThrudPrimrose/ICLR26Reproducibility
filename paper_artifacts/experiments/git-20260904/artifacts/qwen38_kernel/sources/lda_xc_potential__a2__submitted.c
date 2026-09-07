/* LDA XC potential/energy: Slater exchange + Perdew-Zunger correlation.
 * Optimized C implementation. */
#include <stdint.h>
#include <math.h>

#define AX    0.9847450218426965
#define GAMMA (-0.1423)
#define B1    1.0529
#define B2    0.3334
#define CA    0.0311
#define CB    (-0.0480)
#define CC    0.0020
#define CD    (-0.0116)

#define INV3 0.333333333333333333333333333333333333

/* (3/(4*pi))^(1/3) and log(3/(4*pi)) */
#define C3  0.6203504908994001
#define K3  (-1.4324119583011812)

/* folded constants for the rs<1 branch */
#define E1 (CB - CA/3.0)
#define E2 (2.0/3.0*CC)
#define E3 ((2.0*CD - CC)/3.0)
/* folded constants for the rs>=1 branch */
#define F1 ((7.0/6.0)*B1)
#define F2 ((4.0/3.0)*B2)

static void impl(double *restrict rho, double *restrict vxc,
                 double *restrict exc, int64_t N, double dvol)
{
    int64_t total = N * N * N;
    double acc = 0.0;
    int64_t i;
    #pragma omp parallel for schedule(static) reduction(+:acc)
    for (i = 0; i < total; ++i) {
        double r = rho[i];
        double n = (r < 1.0e-12 && r == r) ? 1.0e-12 : r;  /* np.maximum */
        double L   = log(n);
        double n13 = exp(L * INV3);
        double rs  = C3 / n13;
        double lnr = (K3 - L) * INV3;
        double s   = sqrt(rs);
        double denom = 1.0 + B1*s + B2*rs;
        double invd  = 1.0 / denom;
        double eps_ge = GAMMA * invd;
        double vc_ge  = eps_ge * (1.0 + F1*s + F2*rs) * invd;
        double eps_lt = CA*lnr + CB + CC*rs*lnr + CD*rs;
        double vc_lt  = CA*lnr + E1 + E2*rs*lnr + E3*rs;
        int hi = rs < 1.0;
        vxc[i] = -AX*n13 + (hi ? vc_lt : vc_ge);
        acc   += (-0.75*AX*n13 + (hi ? eps_lt : eps_ge)) * n;
    }
    *exc = dvol * acc;
}

/* Primary guess: arrays alphabetical (as auto-generated refs order them),
 * dims, then scalars: (exc, rho, vxc, N, dvol) */
void lda_xc_potential_fp64(double *restrict exc, double *restrict rho,
                           double *restrict vxc, int64_t N, double dvol)
{ impl(rho, vxc, exc, N, dvol); }

/* Alternate guess 1: numpy arg order for arrays: (rho, vxc, exc, N, dvol) */
void lda_xc_potential(double *restrict rho, double *restrict vxc,
                      double *restrict exc, int64_t N, double dvol)
{ impl(rho, vxc, exc, N, dvol); }

/* Alternate guess 2: numpy signature order with dim appended: (dvol, rho, vxc, exc, N) */
void lda_xc_potential_0(double dvol, double *restrict rho, double *restrict vxc,
                        double *restrict exc, int64_t N)
{ impl(rho, vxc, exc, N, dvol); }
