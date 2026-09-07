#define _USE_MATH_DEFINES
#include <complex.h>
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void addusxx_g(
    double complex *restrict rhoc,
    const double complex *restrict becphi_c,
    const double complex *restrict becpsi_c,
    const double *restrict xk,
    const double *restrict xkq,
    const double *restrict tau,
    const long long *restrict ityp,
    const long long *restrict tvanp,
    const long long *restrict nh_type,
    const long long *restrict ofsbeta,
    const long long *restrict nij_type,
    const long long *restrict ijtoh,
    const double complex *restrict qgm,
    const long long *restrict mill,
    const double complex *restrict eigts1,
    const double complex *restrict eigts2,
    const double complex *restrict eigts3,
    const long long *restrict nl,
    long long ngms,
    long long nnr,
    long long nr1,
    long long nr2,
    long long nr3,
    long long nat,
    long long ntyp,
    long long nkb,
    long long nhm,
    long long nij_tot)
{
    const double tpi = 2.0 * M_PI;
    double complex eigqts[nat];

    for (long long na = 0; na < nat; ++na) {
        double arg = tpi * ((xk[0] - xkq[0]) * tau[0 * nat + na] +
                            (xk[1] - xkq[1]) * tau[1 * nat + na] +
                            (xk[2] - xkq[2]) * tau[2 * nat + na]);
        eigqts[na] = cos(arg) - 1.0 * I * sin(arg);
    }

    for (long long nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;
        long long nij = nij_type[nt];
        for (long long na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            long long ijkb0 = ofsbeta[na];
            long long nh = nh_type[nt];
            for (long long ig = 0; ig < ngms; ++ig) {
                double complex aux2 = 0.0 + 0.0 * I;
                for (long long ih = 0; ih < nh; ++ih) {
                    long long ikb = ijkb0 + ih;
                    double complex aux1 = 0.0 + 0.0 * I;
                    for (long long jh = 0; jh < nh; ++jh) {
                        long long jkb = ijkb0 + jh;
                        long long ijh = ijtoh[ih * nhm * ntyp + jh * ntyp + nt];
                        if (ijh < 0) continue;
                        aux1 += qgm[ig * nij_tot + nij + ijh] * becpsi_c[jkb];
                    }
                    aux2 += aux1 * conj(becphi_c[ikb]);
                }
                double complex sf = eigts1[(mill[0 * ngms + ig] + nr1) * nat + na]
                                  * eigts2[(mill[1 * ngms + ig] + nr2) * nat + na]
                                  * eigts3[(mill[2 * ngms + ig] + nr3) * nat + na];
                rhoc[nl[ig]] += aux2 * eigqts[na] * sf;
            }
        }
    }
}
