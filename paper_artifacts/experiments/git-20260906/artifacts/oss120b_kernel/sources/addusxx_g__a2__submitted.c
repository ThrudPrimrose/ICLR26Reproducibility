/*
 * addusxx_g kernel implementation in C.
 * Direct translation of the NumPy reference in addusxx_g_numpy.py.
 * Operates on double‑precision complex numbers (complex128).
 */

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <complex.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Helper for complex conjugation – portable for double complex. */
static inline double _Complex __npb_conj(double _Complex z) {
    return __builtin_complex(__real__ z, -__imag__ z);
}

void addusxx_g_fp64(
    double _Complex * __restrict rhoc,
    const double _Complex * __restrict becphi_c,
    const double _Complex * __restrict becpsi_c,
    const double * __restrict xk,
    const double * __restrict xkq,
    const double * __restrict tau,          // (3, nat) row‑major
    const int64_t * __restrict ityp,
    const int64_t * __restrict tvanp,
    const int64_t * __restrict nh_type,
    const int64_t * __restrict ofsbeta,
    const int64_t * __restrict nij_type,
    const int64_t * __restrict ijtoh,       // (nhm, nhm, ntyp) row‑major
    const double _Complex * __restrict qgm, // (ngms, nij_tot)
    const int64_t * __restrict mill,        // (3, ngms)
    const double _Complex * __restrict eigts1, // (2*nr1+1, nat)
    const double _Complex * __restrict eigts2, // (2*nr2+1, nat)
    const double _Complex * __restrict eigts3, // (2*nr3+1, nat)
    const int64_t * __restrict nl,
    const int64_t ngms,
    const int64_t nnr,
    const int64_t nr1,
    const int64_t nr2,
    const int64_t nr3,
    const int64_t nat,
    const int64_t ntyp,
    const int64_t nkb,
    const int64_t nhm,
    const int64_t nij_tot
) {
    /* -------------------------------------------------------------------- */
    /* 1. Phase factors eigqts[na] = cos(arg) - I*sin(arg) */
    double _Complex * __restrict eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));
    if (!eigqts) return; // allocation failure
    const double tpi = 2.0 * M_PI;
    const double dk0 = xk[0] - xkq[0];
    const double dk1 = xk[1] - xkq[1];
    const double dk2 = xk[2] - xkq[2];
    for (int64_t na = 0; na < nat; ++na) {
        double arg = tpi * (dk0 * tau[0 * nat + na] +
                            dk1 * tau[1 * nat + na] +
                            dk2 * tau[2 * nat + na]);
        eigqts[na] = cos(arg) - I * sin(arg);
    }

    /* -------------------------------------------------------------------- */
    /* 2. Main accumulation loops */
#if defined(_OPENMP)
#pragma omp parallel
{
    double _Complex *local_rhoc = (double _Complex *)calloc((size_t)nnr, sizeof(double _Complex));
    if (!local_rhoc) {
        #pragma omp critical
        { fprintf(stderr, "Memory allocation failed for local_rhoc\n"); abort(); }
    }
    #pragma omp for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        for (int64_t nt = 0; nt < ntyp; ++nt) {
            if (!tvanp[nt]) continue;               // skip non‑ultrasoft species
            const int64_t nij = nij_type[nt];
            int64_t nh = nh_type[nt];
            if (nh > nhm) nh = nhm;
            for (int64_t na = 0; na < nat; ++na) {
                if (ityp[na] != nt) continue;
                const int64_t ijkb0 = ofsbeta[na];
                double _Complex aux2 = 0.0 + 0.0*I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    const int64_t ikb = ijkb0 + ih;
                    if (ikb < 0 || ikb >= nkb) continue;
                    double _Complex aux1 = 0.0 + 0.0*I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        const int64_t jkb = ijkb0 + jh;
                        if (jkb < 0 || jkb >= nkb) continue;
                        const int64_t ijtoh_idx = ((ih * nhm) + jh) * ntyp + nt;
                        int64_t ij = ijtoh[ijtoh_idx];
                        if (ij < 0) continue;
                        const int64_t col = nij + ij;
                        if ((uint64_t)col >= (uint64_t)nij_tot) continue;
                        size_t qgm_offset = (size_t)ig * (size_t)nij_tot + (size_t)col;
                        if (qgm_offset < (size_t)ngms * (size_t)nij_tot) {
                            aux1 += qgm[qgm_offset] * becpsi_c[jkb];
                        }
                    }
                    aux2 += aux1 * __npb_conj(becphi_c[ikb]);
                }
                const int64_t m0 = mill[ig];
                const int64_t m1 = mill[ngms + ig];
                const int64_t m2 = mill[2*ngms + ig];
                const int64_t row0 = m0 + nr1;
                const int64_t row1 = m1 + nr2;
                const int64_t row2 = m2 + nr3;
                size_t idx1 = (size_t)row0 * (size_t)nat + (size_t)na;
                size_t idx2 = (size_t)row1 * (size_t)nat + (size_t)na;
                size_t idx3 = (size_t)row2 * (size_t)nat + (size_t)na;
                double _Complex e1 = eigts1[idx1];
                double _Complex e2 = eigts2[idx2];
                double _Complex e3 = eigts3[idx3];
                aux2 *= eigqts[na] * e1 * e2 * e3;
                int64_t rho_idx = nl[ig];
                if (rho_idx >= 0 && rho_idx < nnr) {
                    local_rhoc[rho_idx] += aux2;
                }
            }
        }
    }
    #pragma omp critical
    {
        for (int64_t i = 0; i < nnr; ++i) {
            rhoc[i] += local_rhoc[i];
        }
    }
    free(local_rhoc);
}
#else

    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;               // skip non‑ultrasoft species
        const int64_t nij = nij_type[nt];        // column offset for this species
        int64_t nh = nh_type[nt];                // number of projectors for this species
        if (nh > nhm) nh = nhm;                  // defensive clamp
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            const int64_t ijkb0 = ofsbeta[na];   // base index for beta functions of atom na
            for (int64_t ig = 0; ig < ngms; ++ig) {
                double _Complex aux2 = 0.0 + 0.0*I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                                    const int64_t ikb = ijkb0 + ih;
                if (ikb < 0 || ikb >= nkb) continue;
                    double _Complex aux1 = 0.0 + 0.0*I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                                        const int64_t jkb = ijkb0 + jh;
                if (jkb < 0 || jkb >= nkb) continue;
                        const int64_t ijtoh_idx = ((ih * nhm) + jh) * ntyp + nt;
                        int64_t ij = ijtoh[ijtoh_idx];
                        if (ij < 0) continue; // unused entry
                        const int64_t col = nij + ij;
                if ((uint64_t)col >= (uint64_t)nij_tot) continue;
                                        size_t qgm_offset = (size_t)ig * (size_t)nij_tot + (size_t)col;
                if (qgm_offset < (size_t)ngms * (size_t)nij_tot) {
                    aux1 += qgm[qgm_offset] * becpsi_c[jkb];
                }
                    }
                    aux2 += aux1 * __npb_conj(becphi_c[ikb]);
                }
                const int64_t m0 = mill[ig];               // mill[0, ig]
                const int64_t m1 = mill[ngms + ig];        // mill[1, ig]
                const int64_t m2 = mill[2*ngms + ig];      // mill[2, ig]
                const int64_t row0 = m0 + nr1;
                const int64_t row1 = m1 + nr2;
                const int64_t row2 = m2 + nr3;
                size_t idx1 = (size_t)row0 * (size_t)nat + (size_t)na;
                double _Complex e1 = eigts1[idx1];
                size_t idx2 = (size_t)row1 * (size_t)nat + (size_t)na;
                double _Complex e2 = eigts2[idx2];
                size_t idx3 = (size_t)row2 * (size_t)nat + (size_t)na;
                double _Complex e3 = eigts3[idx3];
                aux2 *= eigqts[na] * e1 * e2 * e3;
                int64_t rho_idx = nl[ig];
                if (rho_idx >= 0 && rho_idx < nnr) {
                    rhoc[rho_idx] += aux2;
                }
            }
        }
    }
    #endif
#if 0
/* -------------------------------------------------------------------- */
/* 2. Main accumulation loops */
#if defined(_OPENMP)
    /* Parallel over G‑Vectors (ig). Each thread processes a disjoint set of ig,
       avoiding any race on rhoc. */
    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        for (int64_t nt = 0; nt < ntyp; ++nt) {
            if (!tvanp[nt]) continue;               // skip non‑ultrasoft species
            const int64_t nij = nij_type[nt];
            int64_t nh = nh_type[nt];
            if (nh > nhm) nh = nhm;
            for (int64_t na = 0; na < nat; ++na) {
                if (ityp[na] != nt) continue;
                const int64_t ijkb0 = ofsbeta[na];
                double _Complex aux2 = 0.0 + 0.0*I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    const int64_t ikb = ijkb0 + ih;
                    if (ikb < 0 || ikb >= nkb) continue;
                    double _Complex aux1 = 0.0 + 0.0*I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        const int64_t jkb = ijkb0 + jh;
                        if (jkb < 0 || jkb >= nkb) continue;
                        const int64_t ijtoh_idx = ((ih * nhm) + jh) * ntyp + nt;
                        int64_t ij = ijtoh[ijtoh_idx];
                        if (ij < 0) continue;
                        const int64_t col = nij + ij;
                        if ((uint64_t)col >= (uint64_t)nij_tot) continue;
                        size_t qgm_offset = (size_t)ig * (size_t)nij_tot + (size_t)col;
                        if (qgm_offset < (size_t)ngms * (size_t)nij_tot) {
                            aux1 += qgm[qgm_offset] * becpsi_c[jkb];
                        }
                    }
                    aux2 += aux1 * __npb_conj(becphi_c[ikb]);
                }
                /* Phase‑factor accumulation */
                const int64_t m0 = mill[ig];
                const int64_t m1 = mill[ngms + ig];
                const int64_t m2 = mill[2*ngms + ig];
                const int64_t row0 = m0 + nr1;
                const int64_t row1 = m1 + nr2;
                const int64_t row2 = m2 + nr3;
                size_t idx1 = (size_t)row0 * (size_t)nat + (size_t)na;
                size_t idx2 = (size_t)row1 * (size_t)nat + (size_t)na;
                size_t idx3 = (size_t)row2 * (size_t)nat + (size_t)na;
                double _Complex e1 = eigts1[idx1];
                double _Complex e2 = eigts2[idx2];
                double _Complex e3 = eigts3[idx3];
                aux2 *= eigqts[na] * e1 * e2 * e3;
                int64_t rho_idx = nl[ig];
                if (rho_idx >= 0 && rho_idx < nnr) {
                    rhoc[rho_idx] += aux2;
                }
            }
        }
    }
#else
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;               // skip non‑ultrasoft species
        const int64_t nij = nij_type[nt];
        int64_t nh = nh_type[nt];
        if (nh > nhm) nh = nhm;
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            const int64_t ijkb0 = ofsbeta[na];
            for (int64_t ig = 0; ig < ngms; ++ig) {
                double _Complex aux2 = 0.0 + 0.0*I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    const int64_t ikb = ijkb0 + ih;
                    if (ikb < 0 || ikb >= nkb) continue;
                    double _Complex aux1 = 0.0 + 0.0*I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        const int64_t jkb = ijkb0 + jh;
                        if (jkb < 0 || jkb >= nkb) continue;
                        const int64_t ijtoh_idx = ((ih * nhm) + jh) * ntyp + nt;
                        int64_t ij = ijtoh[ijtoh_idx];
                        if (ij < 0) continue;
                        const int64_t col = nij + ij;
                        if ((uint64_t)col >= (uint64_t)nij_tot) continue;
                        size_t qgm_offset = (size_t)ig * (size_t)nij_tot + (size_t)col;
                        if (qgm_offset < (size_t)ngms * (size_t)nij_tot) {
                            aux1 += qgm[qgm_offset] * becpsi_c[jkb];
                        }
                    }
                    aux2 += aux1 * __npb_conj(becphi_c[ikb]);
                }
                const int64_t m0 = mill[ig];
                const int64_t m1 = mill[ngms + ig];
                const int64_t m2 = mill[2*ngms + ig];
                const int64_t row0 = m0 + nr1;
                const int64_t row1 = m1 + nr2;
                const int64_t row2 = m2 + nr3;
                size_t idx1 = (size_t)row0 * (size_t)nat + (size_t)na;
                size_t idx2 = (size_t)row1 * (size_t)nat + (size_t)na;
                size_t idx3 = (size_t)row2 * (size_t)nat + (size_t)na;
                double _Complex e1 = eigts1[idx1];
                double _Complex e2 = eigts2[idx2];
                double _Complex e3 = eigts3[idx3];
                aux2 *= eigqts[na] * e1 * e2 * e3;
                int64_t rho_idx = nl[ig];
                if (rho_idx >= 0 && rho_idx < nnr) {
                    rhoc[rho_idx] += aux2;
                }
            }
        }
    }
#endif
#endif
free(eigqts);
}
