/* addusxx_g optimized v2: symmetry-packed contiguous dot product (explicit
 * real/imag for AVX-512 vectorization) + OpenMP over ig.
 *
 * Per (ig, na): aux2 = sum_i sum_j Q_ij v_j conj(p_i), Q symmetric.
 * = sum_{k=0}^{nh(nh+1)/2-1} row[k] * W[k], where row = qgm[ig, nij+ ..]
 * and W (ig-independent, precomputed per atom) holds in upper-triangle
 * packed order:
 *   W[k(i,i)] = conj(p_i) v_i,  W[k(i,j)] = conj(p_i) v_j + conj(p_j) v_i
 * All atoms summed per ig, single scatter -> race-free parallel ig loop.
 */
#include <stdint.h>
#include <complex.h>
#include <math.h>

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
    (void)ijtoh; (void)nhm; (void)ntyp; (void)nkb; (void)nnr;
    const double *restrict becphi = (const double *)becphi_c;
    const double *restrict becpsi = (const double *)becpsi_c;
    const double *restrict e1r = (const double *)eigts1;
    const double *restrict e2r = (const double *)eigts2;
    const double *restrict e3r = (const double *)eigts3;
    const double *restrict qg = (const double *)qgm;
    double *restrict rh = (double *)rhoc;
    const int64_t nqc = 2 * nij_tot; /* qgm row stride in doubles */

    double tpi = 2.0 * 3.141592653589793;
    const int64_t NHC = 20;

    double ere[64], eim[64];
    for (int64_t na = 0; na < nat; ++na) {
        double s0 = (xk[0] - xkq[0]) * tau[0 * nat + na];
        double s1 = (xk[1] - xkq[1]) * tau[1 * nat + na];
        double s2 = (xk[2] - xkq[2]) * tau[2 * nat + na];
        double arg = tpi * (s0 + s1 + s2);
        double c = cos(arg), sn = sin(arg);
        ere[na] = c; eim[na] = -sn;
    }

    double W[64][(NHC * (NHC + 1)) / 2 * 2]; /* real/imag interleaved */
    const double *Wp[64];
    int64_t nuniq[64];
    int64_t qoff[64];  /* qgm row column offset in doubles */

    for (int64_t na = 0; na < nat; ++na) {
        int64_t nt = ityp[na];
        int64_t nh = nh_type[nt];
        const double *p = becphi + 2 * ofsbeta[na];
        const double *v = becpsi + 2 * ofsbeta[na];
        double *w = W[na];
        int64_t k = 0;
        if (tvanp[nt]) {
            for (int64_t i = 0; i < nh; ++i) {
                double pir = p[2 * i], pii = p[2 * i + 1];
                double vir = v[2 * i], vis = v[2 * i + 1];
                for (int64_t j = i; j < nh; ++j) {
                    double vr = v[2 * j], vi = v[2 * j + 1];
                    double cr, ci;
                    /* W = conj(p_i)*v_j + conj(p_j)*v_i  (i<j);
                       diag (i==j): conj(p_i)*v_i */
                    double t1r = pir * vr + pii * vi;   /* conj(p_i)*v_j */
                    double t1i = pir * vi - pii * vr;
                    double t2r, t2i;
                    if (i == j) {
                        t2r = 0.0; t2i = 0.0;
                    } else {
                        double pjr = p[2 * j], pji = p[2 * j + 1];
                        /* conj(p_j)*v_i */
                        t2r = pjr * vir + pji * vis;
                        t2i = pjr * vis - pji * vir;
                    }
                    cr = t1r + t2r;
                    ci = t1i + t2i;
                    w[2 * k] = cr;
                    w[2 * k + 1] = ci;
                    ++k;
                }
            }
        }
        Wp[na] = W[na];
        qoff[na] = 2 * nij_type[nt];
        nuniq[na] = k;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        int64_t m1 = mill[0 * ngms + ig];
        int64_t m2 = mill[1 * ngms + ig];
        int64_t m3 = mill[2 * ngms + ig];
        int64_t r1 = 2 * ((m1 + nr1) * nat);
        int64_t r2 = 2 * ((m2 + nr2) * nat);
        int64_t r3 = 2 * ((m3 + nr3) * nat);
        double tre = 0.0, ti = 0.0;
        for (int64_t na = 0; na < nat; ++na) {
            int64_t nq = nuniq[na];
            if (!nq) continue;
            const double *restrict row = qg + ig * nqc + qoff[na];
            const double *restrict w = Wp[na];
            double sr = 0.0, si = 0.0;
            for (int64_t k = 0; k < nq; ++k) {
                double ar = row[2 * k], ai = row[2 * k + 1];
                double br = w[2 * k], bi = w[2 * k + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            /* s * eigqts * E1 * E2 * E3 ; eigqts = ere + i*eim (complex) */
            int64_t b = 2 * na;
            double qr = ere[na], qi = eim[na];        /* eigqts (complex) */
            double x1r = e1r[r1 + b], x1i = e1r[r1 + b + 1];
            double y1r = qr * x1r - qi * x1i, y1i = qr * x1i + qi * x1r;
            double x2r = e2r[r2 + b], x2i = e2r[r2 + b + 1];
            double y2r = y1r * x2r - y1i * x2i, y2i = y1r * x2i + y1i * x2r;
            double x3r = e3r[r3 + b], x3i = e3r[r3 + b + 1];
            double y3r = y2r * x3r - y2i * x3i, y3i = y2r * x3i + y2i * x3r;
            tre += sr * y3r - si * y3i;
            ti  += sr * y3i + si * y3r;
        }
        int64_t idx = 2 * nl[ig];
        rh[idx] += tre;
        rh[idx + 1] += ti;
    }
}
