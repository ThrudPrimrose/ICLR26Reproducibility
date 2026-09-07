/* QE us_exx::addusxx_g (flag='c' complex-k branch) -- optimized.
 *
 * Identity used: the reference sums, per (ig, na), over all ordered pairs
 * (ih, jh):  qgm[ig, nij + ijtoh[ih, jh, nt]] * bpsi[jh] * conj(bphi[ih]).
 * The packed symmetric map ijtoh folds (i, j) and (j, i) into ONE qgm column
 * p, so the two ordered pairs that share a column are merged into a single
 * per-atom weight  W[p] = bpsi[j]*conj(bphi[i]) + bpsi[i]*conj(bphi[j])
 * (single term when i == j).  The (ih, jh) nest then collapses to one
 * CONTIGUOUS complex dot of the qgm row segment qgm[ig, nij .. nij+nhp)
 * against W -- no gathers, fully streamable/vectorizable.
 *
 * nl is duplicate-free (QE invariant), so each ig writes one distinct rhoc
 * cell: the ig loop is parallel with no cross-thread reduction.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

/* Complex dot product sum_k a[k] * w[k] (NO conjugation), accumulated as
 * four real partial sums so the vectorizer emits pure FMA reductions.
 *   re = s1 - s2,  im = s3 + s4 */
static inline double _Complex cdot_seg(const double _Complex *restrict a,
                                       const double _Complex *restrict w,
                                       int64_t n)
{
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    for (int64_t k = 0; k < n; k++) {
        s1 += creal(a[k]) * creal(w[k]);
        s2 += cimag(a[k]) * cimag(w[k]);
        s3 += creal(a[k]) * cimag(w[k]);
        s4 += cimag(a[k]) * creal(w[k]);
    }
    return (s1 - s2) + _Complex_I * (s3 + s4);
}

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
                    const int64_t nat, const int64_t ngms, const int64_t nhm,
                    const int64_t nij_tot, const int64_t nkb, const int64_t nnr,
                    const int64_t nr1, const int64_t nr2, const int64_t nr3,
                    const int64_t ntyp)
{
    (void)nkb; (void)nnr;
    const double tpi = 2.0 * M_PI;

    int64_t nmaxp = 1;
    for (int64_t nt = 0; nt < ntyp; nt++) {
        if (tvanp[nt]) {
            int64_t nh = nh_type[nt];
            int64_t p = nh * (nh + 1) / 2;
            if (p > nmaxp) nmaxp = p;
        }
    }

    double _Complex *eigqts = malloc((size_t)nat * sizeof(*eigqts));
    double _Complex *W      = malloc((size_t)nat * (size_t)nmaxp * sizeof(*W));
    int64_t *inv            = malloc(2 * nmaxp * sizeof(*inv)); /* i(p) then j(p) */

    for (int64_t na = 0; na < nat; na++) {
        double arg = tpi * ((xk[0] - xkq[0]) * tau[na]
                          + (xk[1] - xkq[1]) * tau[nat + na]
                          + (xk[2] - xkq[2]) * tau[2 * nat + na]);
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }

    for (int64_t nt = 0; nt < ntyp; nt++) {
        if (!tvanp[nt]) continue;
        int64_t nh = nh_type[nt];
        int64_t nhp = nh * (nh + 1) / 2;
        /* inverse of the packed-symmetric map: column p <-> (i(p), j(p)), i<=j */
        for (int64_t i = 0; i < nh; i++)
            for (int64_t j = i; j < nh; j++) {
                int64_t p = ijtoh[(i * nhm + j) * ntyp + nt];
                inv[p] = i;
                inv[nmaxp + p] = j;
            }
        for (int64_t na = 0; na < nat; na++) {
            if (ityp[na] != nt) continue;
            int64_t b0 = ofsbeta[na];
            double _Complex *w = W + (size_t)na * nmaxp;
            for (int64_t p = 0; p < nhp; p++) {
                int64_t i = inv[p], j = inv[nmaxp + p];
                w[p] = becpsi_c[b0 + j] * conj(becphi_c[b0 + i]);
                if (i != j)
                    w[p] += becpsi_c[b0 + i] * conj(becphi_c[b0 + j]);
            }
        }
    }

    /* per-atom: active flag, segment length, qgm column offset */
    int64_t *ameta = malloc(3 * nat * sizeof(*ameta));
    for (int64_t na = 0; na < nat; na++) {
        int64_t nt = ityp[na];
        ameta[3 * na] = tvanp[nt] ? 1 : 0;
        int64_t nh = nh_type[nt];
        ameta[3 * na + 1] = nh * (nh + 1) / 2;
        ameta[3 * na + 2] = nij_type[nt];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ig++) {
        int64_t l = nl[ig];
        int64_t m0 = mill[ig] + nr1;
        int64_t m1 = mill[ngms + ig] + nr2;
        int64_t m2 = mill[2 * ngms + ig] + nr3;
        double _Complex acc = 0.0;
        for (int64_t na = 0; na < nat; na++) {
            if (!ameta[3 * na]) continue;
            const double _Complex *row = qgm + ig * nij_tot + ameta[3 * na + 2];
            double _Complex t = cdot_seg(row, W + (size_t)na * nmaxp,
                                         ameta[3 * na + 1]);
            t *= eigqts[na];
            t *= eigts1[m0 * nat + na];
            t *= eigts2[m1 * nat + na];
            t *= eigts3[m2 * nat + na];
            acc += t;
        }
        rhoc[l] += acc;
    }

    free(eigqts);
    free(W);
    free(inv);
    free(ameta);
}
