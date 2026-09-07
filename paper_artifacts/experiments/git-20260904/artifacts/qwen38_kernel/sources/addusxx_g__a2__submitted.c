/* addusxx_g -- QE us_exxx augmentation charge, complex-k branch, C (v2 optimized).
 *
 * Per-atom G-dot is folded into a packed-pair B vector once per atom:
 *   aux2(ig,na) = sum_p qgm[ig, col(t)+p] * B_na[p],
 *   B_na[p(ih,jh)] = becpsi[ijkb0+jh] * conj(becphi[ijkb0+ih]),
 * which halves the inner work (nh^2 -> nh(nh+1)/2) and makes it a plain
 * contiguous complex dot that vectorizes.  The ig axis is independent
 * (nl is injective) so it is threaded with OpenMP; every ig is fully
 * reduced inside one thread, so results are run-to-run deterministic.
 */
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <stddef.h>

#define ADDUSXX_MAXNAT 256

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
    int64_t nat, int64_t ngms, int64_t nhm, int64_t nij_tot,
    int64_t nkb, int64_t nnr, int64_t nr1, int64_t nr2, int64_t nr3,
    uint8_t *restrict ws, int64_t ws_bytes)
{
    (void)nkb; (void)nnr; (void)ws; (void)ws_bytes;
    if (nat > ADDUSXX_MAXNAT) nat = ADDUSXX_MAXNAT; /* defensive; never triggers */

    /* species count: all species own at least one atom (ityp dense 0..ntyp-1) */
    int64_t ntyp = 0;
    for (int64_t na = 0; na < nat; na++)
        if (ityp[na] > ntyp) ntyp = ityp[na];
    ntyp += 1;

    const double tpi = 2.0 * 3.141592653589793;

    double _Complex eigqts[ADDUSXX_MAXNAT];
    for (int64_t na = 0; na < nat; na++) {
        double s = (xk[0] - xkq[0]) * tau[na]
                 + (xk[1] - xkq[1]) * tau[nat + na]
                 + (xk[2] - xkq[2]) * tau[2 * nat + na];
        double arg = tpi * s;
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }

    int64_t npmax = nhm * (nhm + 1) / 2;
    int64_t a_col[ADDUSXX_MAXNAT];
    int64_t a_npair[ADDUSXX_MAXNAT];
    char a_act[ADDUSXX_MAXNAT];
    for (int64_t na = 0; na < nat; na++) {
        int64_t t = ityp[na];
        a_act[na] = (tvanp[t] != 0);
        a_col[na] = nij_type[t];
        int64_t nh = nh_type[t];
        a_npair[na] = nh * (nh + 1) / 2;
    }

    /* B[na][p] = becpsi[ijkb0+jh(p)] * conj(becphi[ijkb0+ih(p)]) */
    double _Complex B[(size_t)ADDUSXX_MAXNAT * (size_t)nhm * (size_t)(nhm + 1) / 2];
    for (int64_t na = 0; na < nat; na++) {
        if (!a_act[na]) continue;
        double _Complex *Bna = &B[(size_t)na * npmax];
        int64_t t = ityp[na];
        int64_t ijkb0 = ofsbeta[na];
        int64_t nh = nh_type[t];
        for (int64_t ih = 0; ih < nh; ih++) {
            int64_t ikb = ijkb0 + ih;
            double _Complex cbi = conj(becphi_c[ikb]);
            for (int64_t jh = ih; jh < nh; jh++) {
                int64_t p = ijtoh[(ih * nhm + jh) * ntyp + t];
                double _Complex v = becpsi_c[ijkb0 + jh] * cbi;
                if (jh > ih) v += becpsi_c[ikb] * conj(becphi_c[ijkb0 + jh]);
                Bna[p] = v;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ig++) {
        const double _Complex *e1 = eigts1 + (mill[ig] + nr1) * nat;
        const double _Complex *e2 = eigts2 + (mill[ngms + ig] + nr2) * nat;
        const double _Complex *e3 = eigts3 + (mill[2 * ngms + ig] + nr3) * nat;
        const double _Complex *qrow = qgm + (size_t)ig * nij_tot;
        double _Complex sum = 0.0;
        for (int64_t na = 0; na < nat; na++) {
            if (!a_act[na]) continue;
            const double _Complex *Bna = &B[(size_t)na * npmax];
            int64_t npair = a_npair[na];
            double r = 0.0, im = 0.0;
            #pragma omp simd reduction(+:r,im)
            for (int64_t p = 0; p < npair; p++) {
                double _Complex q = qrow[a_col[na] + p];
                double _Complex b = Bna[p];
                r += creal(q) * creal(b) - cimag(q) * cimag(b);
                im += creal(q) * cimag(b) + cimag(q) * creal(b);
            }
            double _Complex v = (r + _Complex_I * im) * eigqts[na];
            v *= e1[na];
            v *= e2[na];
            v *= e3[na];
            sum += v;
        }
        rhoc[nl[ig]] += sum;
    }
}
