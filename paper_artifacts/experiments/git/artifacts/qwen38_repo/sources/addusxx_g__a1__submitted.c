// Optimized addusxx_g (QE EXX ultrasoft augmentation charge).
//
// Restructure of the naive nt -> na -> ig loop into nt -> ig -> na so that every
// G point is owned by exactly one thread: the accumulation for a given ig runs
// in the same (nt ascending, na ascending) order the reference uses, so each
// per-ig sum matches the reference element-wise, and the final
// rhoc[nl[ig]] += total needs no cross-thread synchronization because nl is
// duplicate-free (the generator asserts it and the reference relies on it).
//
// The per-element order of floating-point operations is unchanged:
//   aux1 = sum_jh qgm[ig, K[ih,jh]] * psi[ofs+jh]      (jh ascending)
//   aux2 = sum_ih aux1_ih * conj(phi[ofs+ih])          (ih ascending)
//   t    = (((aux2 * eigqts[na]) * e1) * e2) * e3
//   total = sum over (nt, na) of t, in reference order
//
// K[ih,jh] = nij_type[nt] + ijtoh[ih,jh,nt] depends only on (nt, ih, jh), so it
// is hoisted out of the ig/na loops (the naive code recomputed the ijtoh index
// for every (na, ig, ih, jh)).

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>

void addusxx_g_fp64(const double _Complex *restrict becphi_c, const double _Complex *restrict becpsi_c, const double _Complex *restrict eigts1, const double _Complex *restrict eigts2, const double _Complex *restrict eigts3, const int64_t *restrict ijtoh, const int64_t *restrict ityp, const int64_t *restrict mill, const int64_t *restrict nh_type, const int64_t *restrict nij_type, const int64_t *restrict nl, const int64_t *restrict ofsbeta, const double _Complex *restrict qgm, double _Complex *restrict rhoc, const double *restrict tau, const int64_t *restrict tvanp, const double *restrict xk, const double *restrict xkq, const int64_t nat, const int64_t ngms, const int64_t nhm, const int64_t nij_tot, const int64_t nkb, const int64_t nnr, const int64_t nr1, const int64_t nr2, const int64_t nr3, const int64_t ntyp) {
    double tpi = 2.0 * 3.141592653589793;

    // --- per-atom phase factor: cos(arg) - i*sin(arg), arg = 2*pi * sum_w (xk[w]-xkq[w])*tau[w,na]
    double _Complex *eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));
    for (int64_t na = 0; na < nat; ++na) {
        double s = 0.0;
        s = (s + (xk[0] - xkq[0]) * tau[0 * nat + na]);
        s = (s + (xk[1] - xkq[1]) * tau[1 * nat + na]);
        s = (s + (xk[2] - xkq[2]) * tau[2 * nat + na]);
        double arg = tpi * s;
        eigqts[na] = (double _Complex)(cos(arg) - sin(arg) * _Complex_I);
    }

    // --- species/atom bookkeeping
    int64_t *sp_natoms = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t *sp_first = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t *sp_nh = (int64_t *)malloc((size_t)ntyp * sizeof(int64_t));
    int64_t *atom_na = (int64_t *)malloc((size_t)nat * sizeof(int64_t));
    int64_t *atom_ofs = (int64_t *)malloc((size_t)nat * sizeof(int64_t));
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        sp_natoms[nt] = 0;
        sp_nh[nt] = tvanp[nt] ? nh_type[nt] : 0;
        sp_first[nt] = 0;
    }
    // pass 1: per-species atom counts and cumulative slot bases
    int64_t base = 0;
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        sp_first[nt] = base;
        int64_t c = 0;
        for (int64_t na = 0; na < nat; ++na)
            if (ityp[na] == nt) c++;
        if (tvanp[nt]) base += c;
        sp_natoms[nt] = tvanp[nt] ? c : 0;
    }
    // pass 2: fill atom lists (ascending na within species == reference order)
    for (int64_t nt = 0; nt < ntyp; ++nt) sp_natoms[nt] = 0;
    for (int64_t na = 0; na < nat; ++na) {
        int64_t nt = ityp[na];
        if (!tvanp[nt]) continue;
        int64_t slot = sp_first[nt] + sp_natoms[nt];
        atom_na[slot] = na;
        atom_ofs[slot] = ofsbeta[na];
        sp_natoms[nt]++;
    }
    // K offsets per species: element offset within a qgm row for each (ih, jh)
    // stored row-major (ih * nh + jh), nh = sp_nh[nt]; packed as int64 (row index).
    // single block, species t starts at t*19*19
    #define KMAX 19
    int64_t *Ktab = (int64_t *)malloc((size_t)ntyp * KMAX * KMAX * sizeof(int64_t));
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        int64_t nh = sp_nh[nt];
        for (int64_t ih = 0; ih < nh; ++ih)
            for (int64_t jh = 0; jh < nh; ++jh)
                Ktab[nt * KMAX * KMAX + ih * KMAX + jh] =
                    nij_type[nt] + ijtoh[(ih * nhm + jh) * ntyp + nt];
    }

    // --- main loop: one thread owns one ig (nl is injective)
    #pragma omp parallel for schedule(dynamic, 32)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        double _Complex tot = 0.0;
        int64_t m0 = mill[ig] + nr1;
        int64_t m1 = mill[ngms + ig] + nr2;
        int64_t m2 = mill[2 * ngms + ig] + nr3;
        const double _Complex *qrow = qgm + ig * nij_tot;
        for (int64_t nt = 0; nt < ntyp; ++nt) {
            int64_t nh = sp_nh[nt];
            if (nh == 0) continue;
            const int64_t *Kt = Ktab + nt * KMAX * KMAX;
            int64_t f = sp_first[nt], n = sp_natoms[nt];
            for (int64_t k = 0; k < n; ++k) {
                int64_t na = atom_na[f + k];
                int64_t ofs = atom_ofs[f + k];
                double _Complex aux2 = 0.0;
                const double _Complex *psi = becpsi_c + ofs;
                const double _Complex *phi = becphi_c + ofs;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    double _Complex aux1 = 0.0;
                    for (int64_t jh = 0; jh < nh; ++jh)
                        aux1 = (aux1 + qrow[Kt[ih * KMAX + jh]] * psi[jh]);
                    aux2 = (aux2 + aux1 * conj(phi[ih]));
                }
                double _Complex t = (aux2 * eigqts[na]);
                t = (t * eigts1[m0 * nat + na]);
                t = (t * eigts2[m1 * nat + na]);
                t = (t * eigts3[m2 * nat + na]);
                tot = (tot + t);
            }
        }
        rhoc[nl[ig]] = (rhoc[nl[ig]] + tot);
    }

    free(Ktab);
    free(atom_ofs);
    free(atom_na);
    free(sp_nh);
    free(sp_first);
    free(sp_natoms);
    free(eigqts);
}
