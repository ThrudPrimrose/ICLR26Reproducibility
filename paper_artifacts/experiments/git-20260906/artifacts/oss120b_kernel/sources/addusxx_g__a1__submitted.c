#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

#ifdef I
#undef I
#endif

void addusxx_g_fp64(double _Complex *restrict rhoc,
                    const double _Complex *restrict becphi_c,
                    const double _Complex *restrict becpsi_c,
                    const double *restrict xk,
                    const double *restrict xkq,
                    const double *restrict tau,
                    const int64_t *restrict ityp,
                    const int64_t *restrict tvanp,
                    const int64_t *restrict nh_type,
                    const int64_t *restrict ofsbeta,
                    const int64_t *restrict nij_type,
                    const int64_t *restrict ijtoh,
                    const double _Complex *restrict qgm,
                    const int64_t *restrict mill,
                    const double _Complex *restrict eigts1,
                    const double _Complex *restrict eigts2,
                    const double _Complex *restrict eigts3,
                    const int64_t *restrict nl,
                    const int64_t ngms,
                    const int64_t nnr,
                    const int64_t nr1,
                    const int64_t nr2,
                    const int64_t nr3,
                    const int64_t nat,
                    const int64_t ntyp,
                    const int64_t nkb,
                    const int64_t nhm,
                    const int64_t nij_tot) {
    const double tpi = 6.28318530717958647692;
    double _Complex *eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));
    if (!eigqts) return;
    for (int64_t na = 0; na < nat; ++na) {
        double arg = tpi * ((xk[0] - xkq[0]) * tau[na] + (xk[1] - xkq[1]) * tau[nat + na] + (xk[2] - xkq[2]) * tau[2*nat + na]);
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (tvanp[nt] == 0) continue;
        int64_t nij = nij_type[nt];
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            int64_t ijkb0 = ofsbeta[na];
            for (int64_t ig = 0; ig < ngms; ++ig) {
                double _Complex aux2 = 0.0 + 0.0*_Complex_I;
                for (int64_t ih = 0; ih < nh_type[nt]; ++ih) {
                    int64_t ikb = ijkb0 + ih;
                    double _Complex aux1 = 0.0 + 0.0*_Complex_I;
                    for (int64_t jh = 0; jh < nh_type[nt]; ++jh) {
                        int64_t jkb = ijkb0 + jh;
                        int64_t ij = ijtoh[(ih * nhm + jh) * ntyp + nt];
                        aux1 += qgm[ig * nij_tot + (nij + ij)] * becpsi_c[jkb];
                    }
                    aux2 += aux1 * conj(becphi_c[ikb]);
                }
                int64_t mh0 = mill[ig];
                int64_t mh1 = mill[ngms + ig];
                int64_t mh2 = mill[2*ngms + ig];
                double _Complex phase = eigqts[na] *
                                         eigts1[(mh0 + nr1) * nat + na] *
                                         eigts2[(mh1 + nr2) * nat + na] *
                                         eigts3[(mh2 + nr3) * nat + na];
                aux2 *= phase;
                rhoc[nl[ig]] += aux2;
            }
        }
    }
    free(eigqts);
}
