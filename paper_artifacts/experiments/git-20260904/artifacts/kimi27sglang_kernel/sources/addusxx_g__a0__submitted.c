#include <stdint.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void addusxx_g_fp64(const double _Complex *restrict becphi_c,
                    const double _Complex *restrict becpsi_c,
                    const double _Complex *restrict eigts1,
                    const double _Complex *restrict eigts2,
                    const double _Complex *restrict eigts3,
                    const int64_t *restrict ijtoh,
                    const int64_t *restrict ityp,
                    const int64_t *restrict mill,
                    const int64_t *restrict nh_type,
                    const int64_t *restrict nl,
                    const int64_t *restrict nij_type,
                    const int64_t *restrict ofsbeta,
                    const double _Complex *restrict qgm,
                    double _Complex *restrict rhoc,
                    const double *restrict tau,
                    const int64_t *restrict tvanp,
                    const double *restrict xk,
                    const double *restrict xkq,
                    int64_t nat,
                    int64_t nhm,
                    int64_t nij_tot,
                    int64_t nkb,
                    int64_t ngms,
                    int64_t nnr,
                    int64_t nr1,
                    int64_t nr2,
                    int64_t nr3,
                    int64_t ntyp,
                    uint8_t *restrict workspace,
                    int64_t workspace_size)
{
    (void)nnr;
    (void)nkb;
    (void)workspace;
    (void)workspace_size;

    const double tpi = 2.0 * M_PI;
    double _Complex *eigqts = (double _Complex *)malloc(nat * sizeof(double _Complex));
    for (int64_t na = 0; na < nat; ++na) {
        double arg = 0.0;
        for (int d = 0; d < 3; ++d) arg += (xk[d] - xkq[d]) * tau[d * nat + na];
        arg *= tpi;
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }

    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;
        int64_t nij = nij_type[nt];
        int64_t nh = nh_type[nt];
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            int64_t ijkb0 = ofsbeta[na];
            for (int64_t ig = 0; ig < ngms; ++ig) {
                double _Complex aux2 = 0.0 + 0.0 * _Complex_I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    int64_t ikb = ijkb0 + ih;
                    double _Complex aux1 = 0.0 + 0.0 * _Complex_I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        int64_t jkb = ijkb0 + jh;
                        int64_t idx = ijtoh[((ih * nhm) + jh) * ntyp + nt];
                        aux1 += qgm[ig * nij_tot + nij + idx] * becpsi_c[jkb];
                    }
                    aux2 += aux1 * conj(becphi_c[ikb]);
                }
                int64_t h1 = mill[ig];
                int64_t h2 = mill[ngms + ig];
                int64_t h3 = mill[2 * ngms + ig];
                aux2 *= eigqts[na]
                      * eigts1[(h1 + nr1) * nat + na]
                      * eigts2[(h2 + nr2) * nat + na]
                      * eigts3[(h3 + nr3) * nat + na];
                rhoc[nl[ig]] += aux2;
            }
        }
    }
    free(eigqts);
}
