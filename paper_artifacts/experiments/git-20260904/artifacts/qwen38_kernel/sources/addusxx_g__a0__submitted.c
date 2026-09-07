/* Optimized QE us_exx::addusxx_g (flag='c' complex k-point branch).
 *
 * Reference math, per atom na of species nt, per G-vector ig:
 *   aux2 = sum_ih [ (sum_jh Q_ig[ih,jh] * psi[ih0+jh]) * conj(phi[ih0+ih]) ]
 *   Q_ig[i,j] = qgm[ig, nij_type[nt] + ijtoh[i,j,nt]]   (symmetric: Q[i,j]=Q[j,i])
 *   rhoc[nl[ig]] += aux2 * eigqts[na] * eigts1[m1+nr1,na] * eigts2[m2+nr2,na] * eigts3[m3+nr3,na]
 *
 * Since Q is symmetric, packing its upper triangle column p(i,j):
 *   aux2 = sum_p q_ig[p] * c_na[p],
 *   c_na[p(i,i)] = psi_i * conj(phi_i)
 *   c_na[p(i,j)] = psi_j * conj(phi_i) + psi_i * conj(phi_j)          (i < j)
 * The c vectors depend ONLY on the atom (not on ig), so they are precomputed once
 * and the per-ig work is a unit-stride complex dot product over the atom's packed
 * qgm columns (nh*(nh+1)/2 instead of nh^2 multiply-adds).
 *
 * The per-ig contributions for all atoms land in the same rhoc cell (nl is
 * injective in ig), so the only cross-atom dependence is a small scalar sum;
 * the whole G dimension is therefore independent and is threaded directly.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef double _Complex cplx;

#define MAXNH 19
#define MAXNPH ((MAXNH) * (MAXNH + 1) / 2) /* 190 */

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
    const int64_t nat,
    const int64_t ngms,
    const int64_t nhm,
    const int64_t nij_tot,
    const int64_t nkb,
    const int64_t nnr,
    const int64_t nr1,
    const int64_t nr2,
    const int64_t nr3,
    const int64_t ntyp,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    (void)nkb;
    (void)nnr;
    (void)workspace;
    (void)workspace_size;

    if (ngms <= 0 || nat <= 0) return;

    const double tpi = 2.0 * 3.141592653589793;

    cplx *const eigqts = (cplx *)malloc((size_t)nat * sizeof(cplx));
    int64_t *const nph_of = (int64_t *)malloc((size_t)nat * sizeof(int64_t));
    int64_t *const qbase_of = (int64_t *)malloc((size_t)nat * sizeof(int64_t));
    cplx *const ctab = (cplx *)malloc((size_t)nat * (size_t)MAXNPH * sizeof(cplx));
    int64_t *const cidx = (int64_t *)malloc((size_t)nat * (size_t)MAXNPH * sizeof(int64_t));

    /* eigqts[na] = exp(-i * tpi * sum_d (xk[d]-xkq[d]) * tau[d,na]) */
    for (int64_t na = 0; na < nat; ++na) {
        const double s = ((xk[0] - xkq[0]) * tau[na]) + ((xk[1] - xkq[1]) * tau[nat + na]) +
                         ((xk[2] - xkq[2]) * tau[2 * nat + na]);
        const double arg = tpi * s;
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }

    /* ijtoh holds SPECIES-LOCAL packed indices: the qgm column of pair p is
     * nij_type[nt] + ijtoh[ih,jh,nt].  Is ijtoh the plain local enumeration 0..nph-1? */
    int gather = 0;
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        const int64_t nh = nh_type[nt];
        int64_t p = 0;
        for (int64_t ih = 0; ih < nh; ++ih)
            for (int64_t jh = ih; jh < nh; ++jh, ++p)
                if (ijtoh[((ih * nhm) + jh) * ntyp + nt] != p) gather = 1;
    }

    /* Per-atom precompute: ctab[na][p], and (gather only) the qgm column of pair p. */
    for (int64_t na = 0; na < nat; ++na) {
        const int64_t nt = ityp[na];
        cplx *const ct = ctab + na * (int64_t)MAXNPH;
        if (!tvanp[nt]) {
            nph_of[na] = 0;
            qbase_of[na] = 0;
            continue;
        }
        const int64_t nh = nh_type[nt];
        const int64_t ofs = ofsbeta[na];
        int64_t p = 0;
        for (int64_t ih = 0; ih < nh; ++ih) {
            const cplx phi_i = becphi_c[ofs + ih];
            const cplx psi_i = becpsi_c[ofs + ih];
            ct[p] = psi_i * conj(phi_i);
            if (gather) cidx[na * (int64_t)MAXNPH + p] = nij_type[nt] + ijtoh[((ih * nhm) + ih) * ntyp + nt];
            ++p;
            for (int64_t jh = ih + 1; jh < nh; ++jh, ++p) {
                const cplx phi_j = becphi_c[ofs + jh];
                const cplx psi_j = becpsi_c[ofs + jh];
                ct[p] = psi_j * conj(phi_i) + psi_i * conj(phi_j);
                if (gather) cidx[na * (int64_t)MAXNPH + p] = nij_type[nt] + ijtoh[((ih * nhm) + jh) * ntyp + nt];
            }
        }
        nph_of[na] = p;
        qbase_of[na] = nij_type[nt];
    }

    if (!gather) {
        #pragma omp parallel for schedule(static)
        for (int64_t ig = 0; ig < ngms; ++ig) {
            const cplx *const qrow = qgm + ig * nij_tot;
            const int64_t m0 = mill[ig] + nr1;
            const int64_t m1 = mill[ngms + ig] + nr2;
            const int64_t m2 = mill[2 * ngms + ig] + nr3;
            cplx s = 0.0;
            for (int64_t na = 0; na < nat; ++na) {
                const int64_t L = nph_of[na];
                if (L <= 0) continue;
                const cplx *const q = qrow + qbase_of[na];
                const cplx *const c = ctab + na * (int64_t)MAXNPH;
                cplx dot = 0.0;
                #pragma omp simd reduction(+:dot)
                for (int64_t p = 0; p < L; ++p) dot += q[p] * c[p];
                const cplx e = eigqts[na] * eigts1[m0 * nat + na] * eigts2[m1 * nat + na] *
                               eigts3[m2 * nat + na];
                s += dot * e;
            }
            rhoc[nl[ig]] += s;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t ig = 0; ig < ngms; ++ig) {
            const cplx *const qrow = qgm + ig * nij_tot;
            const int64_t m0 = mill[ig] + nr1;
            const int64_t m1 = mill[ngms + ig] + nr2;
            const int64_t m2 = mill[2 * ngms + ig] + nr3;
            cplx s = 0.0;
            for (int64_t na = 0; na < nat; ++na) {
                const int64_t L = nph_of[na];
                if (L <= 0) continue;
                const int64_t *const idx = cidx + na * (int64_t)MAXNPH;
                const cplx *const c = ctab + na * (int64_t)MAXNPH;
                cplx dot = 0.0;
                for (int64_t p = 0; p < L; ++p) dot += qrow[idx[p]] * c[p];
                const cplx e = eigqts[na] * eigts1[m0 * nat + na] * eigts2[m1 * nat + na] *
                               eigts3[m2 * nat + na];
                s += dot * e;
            }
            rhoc[nl[ig]] += s;
        }
    }

    free(eigqts);
    free(nph_of);
    free(qbase_of);
    free(ctab);
    free(cidx);
}
