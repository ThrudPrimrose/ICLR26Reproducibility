/* addusxx_g -- QE us_exx ultrasoft augmentation charge, optimized.
 *
 * Strategy vs the naive (nt -> na -> ig -> ih -> jh) scalar nest:
 *   - loop restructure: the outer parallel loop is over G-vectors `ig`; every
 *     ig writes the single, unique cell rhoc[nl[ig]] (nl is duplicate-free),
 *     so there is no cross-thread race and the accumulation into that cell
 *     happens in exactly the reference order (species ascending, then atom
 *     ascending == `na` ascending).
 *   - per (ig, species) the symmetric projector matrix
 *         M[i][j] = qgm[ig, nij_type[s] + ijtoh[i, j, s]]
 *     is unpacked ONCE into a dense nh x nh tile in a per-thread workspace and
 *     reused for every atom of the species (the naive code re-reads the same
 *     qgm entries nh times per atom, nh times per ih).
 *   - aux2 = phi^H M psi is computed as two vectorizable reductions (Mv,
 *     then dot) with -march=native letting GCC use the widest complex SIMD.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double _Complex __npb_conj(double _Complex z) { return conj(z); }

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
    const double tpi = 2.0 * M_PI;
    size_t c16 = sizeof(double _Complex);

    /* ---- eigqts[na] = cos(arg) - i sin(arg), arg = tpi*sum_w (xk-xkq)w * tau[w,na] ---- */
    double _Complex *eigqts = (double _Complex *)malloc((size_t)nat * c16);
    for (int64_t na = 0; na < nat; ++na) {
        double s = ((xk[0] - xkq[0]) * tau[na] +
                    (xk[1] - xkq[1]) * tau[nat + na]) +
                   (xk[2] - xkq[2]) * tau[2 * nat + na];
        double arg = tpi * s;
        eigqts[na] = cos(arg) - _Complex_I * sin(arg);
    }

    /* ---- per-species atom lists (ascending na, as in the reference) ---- */
    int64_t *sp_k = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t tot = 0;
    for (int64_t s = 0; s < ntyp; ++s) {
        int64_t k = 0;
        if (tvanp[s])
            for (int64_t na = 0; na < nat; ++na)
                if (ityp[na] == s) ++k;
        sp_k[s] = k;
        tot += k;
    }
    int64_t *sp_off = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t off = 0;
    for (int64_t s = 0; s < ntyp; ++s) { sp_off[s] = off; off += sp_k[s]; }
    int64_t *sp_na = (int64_t *)malloc(tot ? (size_t)tot * sizeof(int64_t) : sizeof(int64_t));
    int64_t *sp_ob = (int64_t *)malloc(tot ? (size_t)tot * sizeof(int64_t) : sizeof(int64_t));
    int64_t cur = 0;
    for (int64_t s = 0; s < ntyp; ++s)
        for (int64_t na = 0; na < nat; ++na)
            if (tvanp[s] && ityp[na] == s) {
                sp_na[cur] = na;
                sp_ob[cur] = ofsbeta[na];
                ++cur;
            }

    /* ---- qgm column offsets per species, in packed (i, j>=i) order ---- */
    int64_t *sp_npair = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t totcol = 0;
    int64_t maxnh = 0;
    for (int64_t s = 0; s < ntyp; ++s) {
        int64_t nh = tvanp[s] ? nh_type[s] : 0;
        int64_t np = (nh > 0) ? nh * (nh + 1) / 2 : 0;
        sp_npair[s] = np;
        totcol += np;
        if (tvanp[s] && nh > maxnh && sp_k[s] > 0) maxnh = nh;
    }
    int32_t *sp_col = (int32_t *)malloc((totcol ? totcol : 1) * sizeof(int32_t));
    {
        int64_t c = 0;
        for (int64_t s = 0; s < ntyp; ++s) {
            int64_t nh = nh_type[s];
            int64_t nij = nij_type[s];
            for (int64_t i = 0; i < nh; ++i)
                for (int64_t j = i; j < nh; ++j)
                    sp_col[c++] = (int32_t)(nij + ijtoh[(i * nhm + j) * ntyp + s]);
        }
    }
    int64_t *sp_coff = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    off = 0;
    for (int64_t s = 0; s < ntyp; ++s) { sp_coff[s] = off; off += sp_npair[s]; }

    /* per-thread dense M tile (maxnh x maxnh) + w buffer (maxnh) */
    int nthr = omp_get_max_threads();
    if (nthr < 1) nthr = 1;
    double _Complex *mws = (double _Complex *)malloc((size_t)nthr * (size_t)(2 * maxnh * maxnh) * c16);
    double _Complex *wws = mws + (size_t)nthr * (size_t)(maxnh * maxnh);

    /* =================== main loop: one cell of rhoc per ig =================== */
    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        const double _Complex *qrow = qgm + (size_t)ig * (size_t)nij_tot;
        double _Complex acc = rhoc[nl[ig]];
        int64_t m0 = mill[ig] + nr1, m1 = mill[ngms + ig] + nr2, m2 = mill[2 * ngms + ig] + nr3;

        for (int64_t s = 0; s < ntyp; ++s) {
            int64_t k = sp_k[s];
            if (k == 0) continue;
            int64_t nh = nh_type[s];
            const int32_t *cols = sp_col + sp_coff[s];
            const int64_t *na_l = sp_na + sp_off[s];
            const int64_t *ob_l = sp_ob + sp_off[s];

            double _Complex *M = mws + (size_t)omp_get_thread_num() * (size_t)maxnh * maxnh;
            /* unpack packed symmetric tile (p order == (i, j>=i) rows) */
            {
                int64_t p = 0;
                for (int64_t i = 0; i < nh; ++i)
                    for (int64_t j = i; j < nh; ++j) {
                        double _Complex x = qrow[cols[p++]];
                        M[i * nh + j] = x;
                        M[j * nh + i] = x;
                    }
            }

            double _Complex *w = wws + (size_t)omp_get_thread_num() * (size_t)maxnh;
            for (int64_t a = 0; a < k; ++a) {
                const double _Complex *psi = becpsi_c + ob_l[a];
                const double _Complex *phi = becphi_c + ob_l[a];
                int64_t na = na_l[a];

                for (int64_t i = 0; i < nh; ++i) {
                    const double _Complex *mr = M + i * nh;
                    double _Complex a1 = 0.0;
                    for (int64_t j = 0; j < nh; ++j)
                        a1 += mr[j] * psi[j];
                    w[i] = a1;
                }
                double _Complex aux2 = 0.0;
                for (int64_t i = 0; i < nh; ++i)
                    aux2 += w[i] * __npb_conj(phi[i]);

                double _Complex t = aux2 * eigqts[na];
                t *= eigts1[(size_t)m0 * (size_t)nat + na];
                t *= eigts2[(size_t)m1 * (size_t)nat + na];
                t *= eigts3[(size_t)m2 * (size_t)nat + na];
                acc += t;
            }
        }
        rhoc[nl[ig]] = acc;
    }

    free(eigqts); free(sp_k); free(sp_off); free(sp_na); free(sp_ob);
    free(sp_npair); free(sp_col); free(sp_coff); free(mws);
}
