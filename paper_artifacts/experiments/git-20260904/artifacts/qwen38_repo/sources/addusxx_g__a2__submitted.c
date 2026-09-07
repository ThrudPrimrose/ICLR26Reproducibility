// Optimized addusxx_g_fp64: parallelized over G-vectors (ig), with the
// qgm gathers hoisted per (ig, type) and reused across all atoms of that type.
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

void addusxx_g_fp64(const double _Complex *restrict becphi_c, const double _Complex *restrict becpsi_c, const double _Complex *restrict eigts1, const double _Complex *restrict eigts2, const double _Complex *restrict eigts3, const int64_t *restrict ijtoh, const int64_t *restrict ityp, const int64_t *restrict mill, const int64_t *restrict nh_type, const int64_t *restrict nij_type, const int64_t *restrict nl, const int64_t *restrict ofsbeta, const double _Complex *restrict qgm, double _Complex *restrict rhoc, const double *restrict tau, const int64_t *restrict tvanp, const double *restrict xk, const double *restrict xkq, const int64_t nat, const int64_t ngms, const int64_t nhm, const int64_t nij_tot, const int64_t nkb, const int64_t nnr, const int64_t nr1, const int64_t nr2, const int64_t nr3, const int64_t ntyp) {
    (void)nkb;
    (void)nnr;
    const double tpi = 2.0 * 3.141592653589793;

    /* per-atom phase factor: conj(exp(i * tpi * dot(xk-xkq, tau[:,na]))) */
    double _Complex *eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));

    /* group atoms by active type */
    int64_t *nna = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t *atbase = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t *atlist = (int64_t *)malloc((size_t)nat * sizeof(int64_t));
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        atbase[nt] = 0;
        nna[nt] = 0;
    }
    for (int64_t na = 0; na < nat; ++na) {
        int64_t nt = ityp[na];
        if (tvanp[nt]) nna[nt]++;
    }
    for (int64_t nt = 0; nt + 1 < ntyp; ++nt) atbase[nt + 1] = atbase[nt] + nna[nt];
    for (int64_t nt = 0; nt < ntyp; ++nt) nna[nt] = 0;
    for (int64_t na = 0; na < nat; ++na) {
        int64_t nt = ityp[na];
        if (tvanp[nt]) atlist[atbase[nt] + nna[nt]++] = na;
    }

    /* flattened qgm column index per (type, ih, jh): nij_type[nt] + ijtoh[ih, jh, nt] */
    int64_t *qoff = (int64_t *)malloc((size_t)ntyp * (size_t)nhm * (size_t)nhm * sizeof(int64_t));
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;
        int64_t K = nh_type[nt];
        int64_t *qp = qoff + (size_t)nt * (size_t)nhm * (size_t)nhm;
        for (int64_t ih = 0; ih < K; ++ih)
            for (int64_t jh = 0; jh < K; ++jh)
                qp[ih * nhm + jh] = nij_type[nt] + ijtoh[(ih * nhm + jh) * ntyp + nt];
    }

    for (int64_t na = 0; na < nat; ++na) {
        double arg = (xk[0] - xkq[0]) * tau[0 * nat + na] + (xk[1] - xkq[1]) * tau[1 * nat + na] + (xk[2] - xkq[2]) * tau[2 * nat + na];
        double ph = tpi * arg;
        eigqts[na] = cos(ph) - sin(ph) * _Complex_I;
    }

    /* main loop: race-free because nl[ig] is unique per ig */
    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        const double _Complex *qrow = qgm + (size_t)ig * nij_tot;
        const int64_t m1 = mill[ig] + nr1;
        const int64_t m2 = mill[ngms + ig] + nr2;
        const int64_t m3 = mill[2 * ngms + ig] + nr3;
        double _Complex acc = 0.0;
        for (int64_t nt = 0; nt < ntyp; ++nt) {
            if (!tvanp[nt]) continue;
            const int64_t K = nh_type[nt];
            const int64_t *qp = qoff + (size_t)nt * (size_t)nhm * (size_t)nhm;
            double _Complex qm_t[K * K];
            double _Complex S[K];
            double _Complex P[K];
            for (int64_t ih = 0; ih < K; ++ih)
                for (int64_t jh = 0; jh < K; ++jh)
                    qm_t[ih * K + jh] = qrow[qp[ih * nhm + jh]];
            for (int64_t a = 0; a < nna[nt]; ++a) {
                const int64_t na = atlist[atbase[nt] + a];
                const int64_t ofs = ofsbeta[na];
                for (int64_t h = 0; h < K; ++h) {
                    S[h] = becpsi_c[ofs + h];
                    P[h] = conj(becphi_c[ofs + h]);
                }
                double _Complex aux2 = 0.0;
                for (int64_t ih = 0; ih < K; ++ih) {
                    double _Complex t = 0.0;
                    for (int64_t jh = 0; jh < K; ++jh)
                        t = t + qm_t[ih * K + jh] * S[jh];
                    aux2 = aux2 + t * P[ih];
                }
                double _Complex w = eigqts[na] * eigts1[m1 * nat + na] * eigts2[m2 * nat + na] * eigts3[m3 * nat + na];
                acc = acc + aux2 * w;
            }
        }
        rhoc[nl[ig]] = rhoc[nl[ig]] + acc;
    }
    free(qoff);
    free(atlist);
    free(atbase);
    free(nna);
    free(eigqts);
}
