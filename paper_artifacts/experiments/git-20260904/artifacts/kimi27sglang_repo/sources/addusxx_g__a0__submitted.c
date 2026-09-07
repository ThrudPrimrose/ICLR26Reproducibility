#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

void addusxx_g_fp64(const double _Complex *restrict becphi_c,
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
                    const int64_t nat,
                    const int64_t ngms,
                    const int64_t nhm,
                    const int64_t nij_tot,
                    const int64_t nkb,
                    const int64_t nnr,
                    const int64_t nr1,
                    const int64_t nr2,
                    const int64_t nr3,
                    const int64_t ntyp) {
    (void)nkb;
    (void)nnr;

    const double tpi = 6.28318530717958647692;
    double _Complex *eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));
    if (!eigqts) return;

    double dx[3];
    for (int d = 0; d < 3; ++d) dx[d] = xk[d] - xkq[d];

    for (int64_t na = 0; na < nat; ++na) {
        double s = 0.0;
        for (int d = 0; d < 3; ++d) s += dx[d] * tau[d * nat + na];
        double arg = tpi * s;
        eigqts[na] = cos(arg) - (_Complex_I * sin(arg));
    }

    /* Find the largest projector-block size and the largest number of atoms of
       a single ultrasoft species; these determine the temporary buffers. */
    int64_t max_m = 1;
    int64_t max_nij = 1;
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;
        int64_t nh = nh_type[nt];
        int64_t nij = nh * (nh + 1) / 2;
        if (nij > max_nij) max_nij = nij;
        int64_t cnt = 0;
        for (int64_t na = 0; na < nat; ++na)
            if (ityp[na] == nt) ++cnt;
        if (cnt > max_m) max_m = cnt;
    }

    double _Complex *C = (double _Complex *)malloc((size_t)max_nij * max_m * sizeof(double _Complex));
    double _Complex *R = (double _Complex *)malloc((size_t)ngms * max_m * sizeof(double _Complex));
    int64_t *atoms = (int64_t *)malloc((size_t)max_m * sizeof(int64_t));
    if (!C || !R || !atoms) {
        free(eigqts);
        free(C);
        free(R);
        free(atoms);
        return;
    }

    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;

        const int64_t nh = nh_type[nt];
        const int64_t nij_size = nh * (nh + 1) / 2;
        const int64_t base = nij_type[nt];

        int64_t m = 0;
        for (int64_t na = 0; na < nat; ++na)
            if (ityp[na] == nt) atoms[m++] = na;
        if (m == 0) continue;

        /* Build the packed coefficient matrix C[idx * m + a] =
             sum_{ih,jh mapping to idx} becpsi_c[ofsbeta[na]+jh] * conj(becphi_c[ofsbeta[na]+ih]).
           For off-diagonal pairs the two (ih,jh) and (jh,ih) contributions are added here. */
        memset(C, 0, (size_t)nij_size * m * sizeof(double _Complex));
        for (int64_t a = 0; a < m; ++a) {
            const int64_t na = atoms[a];
            const int64_t ijkb0 = ofsbeta[na];
            double _Complex *Ca = &C[a];
            for (int64_t ih = 0; ih < nh; ++ih) {
                const double _Complex ph = conj(becphi_c[ijkb0 + ih]);
                for (int64_t jh = 0; jh < nh; ++jh) {
                    const int64_t idx = ijtoh[((ih * nhm) + jh) * ntyp + nt];
                    if (idx < 0) continue;
                    Ca[idx * m] += becpsi_c[ijkb0 + jh] * ph;
                }
            }
        }

        /* R[ig * m + a] = sum_{packed idx} qgm[ig, base+idx] * C[idx, a],
           then multiply by the atomic phase and scatter-add into rhoc[nl[ig]].
           Parallelising over ig is safe because nl[] is duplicate-free. */
        #pragma omp parallel for schedule(static)
        for (int64_t ig = 0; ig < ngms; ++ig) {
            double _Complex *Rig = &R[ig * m];
            for (int64_t a = 0; a < m; ++a) Rig[a] = 0.0;

            const double _Complex *qrow = &qgm[ig * nij_tot + base];
            for (int64_t idx = 0; idx < nij_size; ++idx) {
                const double _Complex qv = qrow[idx];
                const double _Complex *cvec = &C[idx * m];
                for (int64_t a = 0; a < m; ++a)
                    Rig[a] += qv * cvec[a];
            }

            const int64_t nl_ig = nl[ig];
            const int64_t h1 = mill[0 * ngms + ig] + nr1;
            const int64_t h2 = mill[1 * ngms + ig] + nr2;
            const int64_t h3 = mill[2 * ngms + ig] + nr3;
            for (int64_t a = 0; a < m; ++a) {
                const int64_t na = atoms[a];
                const double _Complex phase = eigqts[na]
                                            * eigts1[h1 * nat + na]
                                            * eigts2[h2 * nat + na]
                                            * eigts3[h3 * nat + na];
                rhoc[nl_ig] += Rig[a] * phase;
            }
        }
    }

    free(eigqts);
    free(C);
    free(R);
    free(atoms);
}
