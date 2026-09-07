#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void addusxx_g_fp64(
    const double _Complex *restrict becphi_c,
    const double _Complex *restrict becpsi_c,
    const double _Complex *restrict eigts1,
    const double _Complex *restrict eigts2,
    const double _Complex *restrict eigts3,
    const int64_t *restrict ijtoh,
    const int64_t *restrict ityp,
    const int64_t *restrict mill,
    const int64_t *restrict nh_type,
    const int64_t *restrict nij_type,
    const int64_t *restrict nl,
    const int64_t *restrict ofsbeta,
    const double _Complex *restrict qgm,
    double _Complex *restrict rhoc,
    const double *restrict tau,
    const int64_t *restrict tvanp,
    const double *restrict xk,
    const double *restrict xkq,
    int64_t nat,
    int64_t ngms,
    int64_t nhm,
    int64_t nij_tot,
    int64_t nkb,
    int64_t nnr,
    int64_t nr1,
    int64_t nr2,
    int64_t nr3,
    int64_t ntyp,
    uint8_t *restrict workspace,
    int64_t workspace_bytes)
{
    (void)nkb;
    (void)nnr;
    (void)workspace;
    (void)workspace_bytes;

    const double tpi = 2.0 * acos(-1.0);
    double _Complex eigqts[nat];

    for (int64_t na = 0; na < nat; ++na) {
        double s = 0.0;
        for (int d = 0; d < 3; ++d) {
            s += (xk[d] - xkq[d]) * tau[d * nat + na];
        }
        double arg = tpi * s;
        eigqts[na] = cos(arg) - 1.0 * I * sin(arg);
    }

    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        int64_t m0 = mill[0 * ngms + ig] + nr1;
        int64_t m1 = mill[1 * ngms + ig] + nr2;
        int64_t m2 = mill[2 * ngms + ig] + nr3;

        double _Complex acc = rhoc[nl[ig]];

        for (int64_t nt = 0; nt < ntyp; ++nt) {
            if (!tvanp[nt]) continue;
            int64_t nij = nij_type[nt];
            int64_t nh = nh_type[nt];

            double _Complex qd[32 * 32];
            for (int64_t ih = 0; ih < nh; ++ih) {
                for (int64_t jh = 0; jh < nh; ++jh) {
                    int64_t ijh = ijtoh[(ih * nhm + jh) * ntyp + nt];
                    qd[ih * 32 + jh] = qgm[ig * nij_tot + nij + ijh];
                }
            }

            for (int64_t na = 0; na < nat; ++na) {
                if (ityp[na] != nt) continue;
                int64_t ijkb0 = ofsbeta[na];

                double _Complex aux1[32];
                for (int64_t ih = 0; ih < nh; ++ih) {
                    double _Complex s = 0.0 + 0.0 * I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        s += qd[ih * 32 + jh] * becpsi_c[ijkb0 + jh];
                    }
                    aux1[ih] = s;
                }

                double _Complex aux2 = 0.0 + 0.0 * I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    aux2 += aux1[ih] * conj(becphi_c[ijkb0 + ih]);
                }
                aux2 = aux2 * eigqts[na]
                     * eigts1[m0 * nat + na]
                     * eigts2[m1 * nat + na]
                     * eigts3[m2 * nat + na];
                acc += aux2;
            }
        }

        rhoc[nl[ig]] = acc;
    }
}
