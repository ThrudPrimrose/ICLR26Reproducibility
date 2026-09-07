/*
 * addusxx_g kernel implementation in C.
 * This is a direct translation of the NumPy reference in addusxx_g_numpy.py.
 * It operates on double-precision complex numbers (complex128) – the default datatype
 * used by the benchmark. The function name follows the convention used in other
 * reference kernels (kernelname_fp64).
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

/*
 * Signature mirrors the order of arguments produced by initialize().
 * All pointer arguments are marked __restrict__ to help the optimizer.
 */
void addusxx_g_fp64(
    double _Complex * __restrict rhoc,
    const double _Complex * __restrict becphi_c,
    const double _Complex * __restrict becpsi_c,
    const double * __restrict xk,
    const double * __restrict xkq,
    const double * __restrict tau,          // shape (3, nat), row‑major
    const int64_t * __restrict ityp,
    const int64_t * __restrict tvanp,
    const int64_t * __restrict nh_type,
    const int64_t * __restrict ofsbeta,
    const int64_t * __restrict nij_type,
    const int64_t * __restrict ijtoh,       // shape (nhm, nhm, ntyp)
    const double _Complex * __restrict qgm, // shape (ngms, nij_tot)
    const int64_t * __restrict mill,        // shape (3, ngms)
    const double _Complex * __restrict eigts1, // shape (2*nr1+1, nat)
    const double _Complex * __restrict eigts2, // shape (2*nr2+1, nat)
    const double _Complex * __restrict eigts3, // shape (2*nr3+1, nat)
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
    /* 1. Compute eigqts[na] = cos(arg) - I*sin(arg) where
     *    arg = 2π * sum_d (xk[d] - xkq[d]) * tau[d,na]
     */
    double _Complex * __restrict eigqts = (double _Complex *)malloc((size_t)nat * sizeof(double _Complex));
    if (!eigqts) return; // allocation failure – nothing to do
    const double tpi = 2.0 * M_PI;
    const double dk0 = xk[0] - xkq[0];
    const double dk1 = xk[1] - xkq[1];
    const double dk2 = xk[2] - xkq[2];
    #if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
for (int64_t na = 0; na < nat; ++na) {
        double arg = tpi * (dk0 * tau[0 * nat + na] +
                            dk1 * tau[1 * nat + na] +
                            dk2 * tau[2 * nat + na]);
        eigqts[na] = cos(arg) - I * sin(arg);
    }

    /* -------------------------------------------------------------------- */
    /* 2. Main accumulation loops */
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;               // skip if species not ultrasoft
        const int64_t nij = nij_type[nt];
        const int64_t nh = nh_type[nt];
        #if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            const int64_t ijkb0 = ofsbeta[na];   // base index for beta functions of atom na
            for (int64_t ig = 0; ig < ngms; ++ig) {
                double _Complex aux2 = 0.0 + 0.0*I;
                for (int64_t ih = 0; ih < nh; ++ih) {
                    const int64_t ikb = ijkb0 + ih;
                    double _Complex aux1 = 0.0 + 0.0*I;
                    for (int64_t jh = 0; jh < nh; ++jh) {
                        const int64_t jkb = ijkb0 + jh;
                        /* ijtoh index = ((ih * nhm) + jh) * ntyp + nt */
                        const int64_t ijtoh_idx = ((ih * nhm) + jh) * ntyp + nt;
                        const int64_t col = nij + ijtoh[ijtoh_idx];
                        if (col < 0 || col >= nij_tot) {
                            fprintf(stderr, "invalid col %ld at ig %ld ih %ld jh %ld nt %ld\n", (long)col, (long)ig, (long)ih, (long)jh, (long)nt);
                            abort();
                        }
                        aux1 += qgm[ig * nij_tot + col] * becpsi_c[jkb];
                    }
                    aux2 += aux1 * __npb_conj(becphi_c[ikb]);
                }
                /* Multiply by phase factors */
                const int64_t m0 = mill[ig];               // mill[0, ig]
                const int64_t m1 = mill[ngms + ig];        // mill[1, ig]
                const int64_t m2 = mill[2*ngms + ig];      // mill[2, ig]
                const int64_t row0 = m0 + nr1;
                const int64_t row1 = m1 + nr2;
                const int64_t row2 = m2 + nr3;
                double _Complex e1 = eigts1[row0 * nat + na];
                double _Complex e2 = eigts2[row1 * nat + na];
                double _Complex e3 = eigts3[row2 * nat + na];
                aux2 *= eigqts[na] * e1 * e2 * e3;
                #pragma omp critical
                {
                    int64_t rho_idx = nl[ig];
                    if (rho_idx >= 0 && rho_idx < nnr) {
                        rhoc[rho_idx] += aux2;
                    }
                }
            }
        }
    }
    free(eigqts);
}
