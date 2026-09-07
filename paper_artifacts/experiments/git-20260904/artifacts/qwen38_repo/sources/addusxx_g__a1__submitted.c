// Optimized QE us_exx::addusxx_g (flag='c' complex-k branch).
//
// The naive reference computes, per (nt, na, ig):
//   aux2 = sum_ih sum_jh qgm[ig, nij + ijtoh[ih,jh,nt]] * becpsi[ih..] * conj(becphi[.])
// and scatters aux2 * eigqts * eigts1 * eigts2 * eigts3 into rhoc[nl[ig]].
//
// Observation: the inner double sum only depends on (ig, na) through qgm, and
//
//   sum_{(ih,jh)} qgm[ig, col(ih,jh)] * conj(phi[ih]) * psi[jh]
//     = sum_col  qgm[ig, col] * w[col],        col = packed pair index,
//   where w[col(ih,jh)] = conj(phi[ih])*psi[jh] + conj(phi[jh])*psi[ih]
//   depends on the atom but NOT on ig (it uses only becphi/becpsi).
//
// So: precompute w per atom once (O(nat*nh^2)); then for every ig compute one
// complex dot product per atom (vectorized, FMA) and scale by
// eigqts[na]*eigts1*eigts2*eigts3.  The G dimension is fully independent
// (nl is injective), so the ig loop is race-free parallel with a plain
// `omp parallel for`.

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

typedef double _Complex cplx;

/* glibc malloc returns 16-byte-aligned blocks; AVX-512 tolerates unaligned
 * loads, so a plain malloc is sufficient for the scratch buffers. */
static void *xalloc(size_t n) { return malloc(n ? n : 16); }

void addusxx_g_fp64(const double _Complex *restrict becphi_c, const double _Complex *restrict becpsi_c,
                    const double _Complex *restrict eigts1, const double _Complex *restrict eigts2,
                    const double _Complex *restrict eigts3, const int64_t *restrict ijtoh,
                    const int64_t *restrict ityp, const int64_t *restrict mill, const int64_t *restrict nh_type,
                    const int64_t *restrict nij_type, const int64_t *restrict nl, const int64_t *restrict ofsbeta,
                    const double _Complex *restrict qgm, double _Complex *restrict rhoc,
                    const double *restrict tau, const int64_t *restrict tvanp, const double *restrict xk,
                    const double *restrict xkq, const int64_t nat, const int64_t ngms, const int64_t nhm,
                    const int64_t nij_tot, const int64_t nkb, const int64_t nnr, const int64_t nr1,
                    const int64_t nr2, const int64_t nr3, const int64_t ntyp) {
    (void)nkb;
    (void)nnr;
    if (nat <= 0 || ngms <= 0 || ntyp <= 0 || nij_tot <= 0) return;

    const double tpi = 2.0 * 3.141592653589793;

    /* ---------- eigqts[na] = exp(-i*tpi*sum_d (xk_d - xkq_d)*tau[d,na]) ---------- */
    double *eqre = (double *)xalloc(nat * 2 * sizeof(double));
    double *eqim = eqre + nat;
    const double dx0 = xk[0] - xkq[0];
    const double dx1 = xk[1] - xkq[1];
    const double dx2 = xk[2] - xkq[2];
    for (int64_t na = 0; na < nat; na++) {
        const double arg = tpi * ((dx0 * tau[na]) + (dx1 * tau[nat + na]) + (dx2 * tau[2 * nat + na]));
        eqre[na] = cos(arg);
        eqim[na] = -sin(arg);
    }

    /* ---------- per-species packed column count and offsets ---------- */
    int64_t *pack2 = (int64_t *)xalloc((ntyp + 1) * sizeof(int64_t)); /* 2*npack per species, cumulative in pack2[ntyp+1]? no: separate */
    int64_t *qoff = (int64_t *)xalloc(ntyp * sizeof(int64_t));        /* qgm column offset per species */
    int64_t *wofs2 = (int64_t *)xalloc((nat + 1) * sizeof(int64_t));  /* w offset per atom (2*npack stride), cumulative */
    int64_t *ityp_l = (int64_t *)xalloc(nat * sizeof(int64_t));
    int64_t wtot2 = 0;
    for (int64_t nt = 0; nt < ntyp; nt++) {
        const int64_t nh = tvanp[nt] ? nh_type[nt] : 0;
        const int64_t p2 = nh > 0 ? 2 * nh * (nh + 1) / 2 : 0;
        pack2[nt] = p2;
        qoff[nt] = nij_type[nt];
    }
    for (int64_t na = 0; na < nat; na++) {
        ityp_l[na] = ityp[na];
        wofs2[na] = wtot2;
        wtot2 += pack2[ityp[na]];
    }
    wofs2[nat] = wtot2;
    double *wbuf = (double *)xalloc((size_t)(wtot2 ? wtot2 : 2) * sizeof(double));

    /* ---------- per-atom weights: w[col] = conj(phi_ih)*psi_jh + conj(phi_jh)*psi_ih ---------- */
    const double *phib = (const double *)becphi_c;
    const double *psib = (const double *)becpsi_c;
    for (int64_t na = 0; na < nat; na++) {
        const int64_t nt = ityp_l[na];
        const int64_t nh = tvanp[nt] ? nh_type[nt] : 0;
        if (nh <= 0) continue;
        const int64_t base = ofsbeta[na];
        double *wa = wbuf + wofs2[na];
        for (int64_t ih = 0; ih < nh; ih++) {
            const double phi_r = phib[2 * (base + ih)];
            const double phi_i = phib[2 * (base + ih) + 1];
            for (int64_t jh = ih; jh < nh; jh++) {
                const int64_t col = ijtoh[(ih * nhm + jh) * ntyp + nt];
                /* conj(phi_ih) * psi_jh : (pr - pi i)(sr + si i) = (pr sr + pi si) + i(pr si - pi sr) */
                const double psi_r = psib[2 * (base + jh)];
                const double psi_i = psib[2 * (base + jh) + 1];
                double vr = fma(phi_i, psi_i, phi_r * psi_r);
                double vi = fma(phi_r, psi_i, -phi_i * psi_r);
                if (jh > ih) {
                    /* + conj(phi_jh) * psi_ih */
                    const double qj_r = phib[2 * (base + jh)];
                    const double qj_i = phib[2 * (base + jh) + 1];
                    const double pis_r = psib[2 * (base + ih)];
                    const double pis_i = psib[2 * (base + ih) + 1];
                    vr = fma(qj_i, pis_i, fma(qj_r, pis_r, vr));
                    vi = fma(qj_r, pis_i, vi - qj_i * pis_r);
                }
                wa[2 * col] = vr;
                wa[2 * col + 1] = vi;
            }
        }
    }

    /* ---------- hot loop: parallel over ig, race-free (nl injective) ---------- */
    const double *e1b = (const double *)eigts1;
    const double *e2b = (const double *)eigts2;
    const double *e3b = (const double *)eigts3;
    const double *qgmb = (const double *)qgm;
    double *rhocb = (double *)rhoc;

#pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ig++) {
        const double *qrow = qgmb + (size_t)ig * (2 * nij_tot);
        const int64_t m0 = mill[ig] + nr1;
        const int64_t m1 = mill[ngms + ig] + nr2;
        const int64_t m2 = mill[2 * ngms + ig] + nr3;
        double ar = 0.0, ai = 0.0;
        for (int64_t na = 0; na < nat; na++) {
            const int64_t p2 = pack2[ityp_l[na]];
            if (p2 == 0) continue;
            const double *p1 = e1b + ((size_t)m0 * nat + na) * 2;
            const double *p2_ = e2b + ((size_t)m1 * nat + na) * 2;
            const double *p3 = e3b + ((size_t)m2 * nat + na) * 2;
            double r = fma(-eqim[na], p1[1], eqre[na] * p1[0]);
            double i = fma(eqre[na], p1[1], eqim[na] * p1[0]);
            double r2 = fma(-i, p2_[1], r * p2_[0]);
            double i2 = fma(r, p2_[1], i * p2_[0]);
            const double cr = fma(-i2, p3[1], r2 * p3[0]);
            const double ci = fma(r2, p3[1], i2 * p3[0]);

            const double *qa = qrow + 2 * qoff[ityp_l[na]];
            const double *wa = wbuf + wofs2[na];
            double dr = 0.0, di = 0.0;
            for (int64_t c = 0; c < p2; c += 2) {
                /* plain ops: -ffp-contract (default fast) emits FMAs and the
                 * loop vectorizes; a spelled-out nested fma() pattern does
                 * not (GCC refuses a vectype for the inner load). */
                dr += qa[c] * wa[c] - qa[c + 1] * wa[c + 1];
                di += qa[c + 1] * wa[c] + qa[c] * wa[c + 1];
            }
            ar += cr * dr - ci * di;
            ai += cr * di + ci * dr;
        }
        double *rc = rhocb + (size_t)nl[ig] * 2;
        rc[0] += ar;
        rc[1] += ai;
    }

    free(eqre);
    free(pack2);
    free(qoff);
    free(wofs2);
    free(ityp_l);
    free(wbuf);
}
