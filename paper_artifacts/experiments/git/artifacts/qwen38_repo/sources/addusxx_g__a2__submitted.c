/* Optimized addusxx_g (QE us_exx::addusxx_g, flag='c' branch).
 *
 * Algebra used: for an atom na of species nt the per-G contribution is
 *
 *   aux2(ig) = sum_ih sum_jh qgm[ig, s + pack(ih,jh)] * psi[jh] * conj(phi[ih])
 *
 * and, since pack(ih,jh) == pack(jh,ih), this is a plain dot product of the
 * qgm row segment with a per-atom weight vector
 *
 *   W[c] = conj(phi[i]) * psi[j] + (i != j) * conj(phi[j]) * psi[i],  c = pack(i,j), i <= j
 *
 * which is precomputed once per atom (it is independent of ig).  The main
 * work is then, per G point, a batch of short dense dot products followed by
 * the phase multiplication and the scatter into rhoc[nl[ig]].  The G loop is
 * embarrassingly parallel (nl is duplicate-free, so distinct ig touch
 * distinct output cells) and the dot products are SIMD (AVX-512 / AVX2,
 * scalar fallback).
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <immintrin.h>

typedef struct {
    int64_t na;      /* atom index */
    int64_t s;       /* column offset of the species block inside qgm rows */
    int nbh;         /* number of packed (i,j) columns of the species */
    const double *w; /* W coefficients (interleaved re/im), packed layout */
    const double *eq;/* eigqts[na] */
} act_t;

#if defined(__AVX512F__)
static inline void dot_cz(const double *restrict a, const double *restrict b, int n,
                          double *re, double *im) {
    /* swap re/im inside each complex: vpermpd picks lane i = x[idx64[i] & 7] */
    static const int64_t sw_idx[8] = {1, 0, 3, 2, 5, 4, 7, 6};
    __m512d sre = _mm512_setzero_pd();
    __m512d sim = _mm512_setzero_pd();
    const __m512i sw = _mm512_loadu_si512(sw_idx);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        __m512d va = _mm512_loadu_pd(a + 2 * k);
        __m512d vb = _mm512_loadu_pd(b + 2 * k);
        __m512d t0 = _mm512_mul_pd(va, vb); /* even: ar*br, odd: ai*bi */
        __m512d t1 = _mm512_mul_pd(_mm512_permutexvar_pd(sw, va), vb); /* even: ai*br, odd: ar*bi */
        sre = _mm512_add_pd(sre, _mm512_sub_pd(t0, _mm512_permutexvar_pd(sw, t0)));
        sim = _mm512_add_pd(sim, _mm512_add_pd(t1, _mm512_permutexvar_pd(sw, t1)));
    }
    for (; k < n; k++) {
        double ar = a[2 * k], ai = a[2 * k + 1];
        double br = b[2 * k], bi = b[2 * k + 1];
        *re += ar * br - ai * bi;
        *im += ar * bi + ai * br;
    }
    /* even lanes hold the partial sums (odd lanes are +/- the same); reduce even lanes */
    sre = _mm512_mask_blend_pd(0x55, _mm512_setzero_pd(), sre);
    *re += _mm512_reduce_add_pd(sre);
    sim = _mm512_mask_blend_pd(0x55, _mm512_setzero_pd(), sim);
    *im += _mm512_reduce_add_pd(sim);
}
#elif defined(__AVX2__)
static inline void dot_cz(const double *restrict a, const double *restrict b, int n,
                          double *re, double *im) {
    static const int64_t sw_idx[4] = {1, 0, 3, 2};
    __m256d sre = _mm256_setzero_pd();
    __m256d sim = _mm256_setzero_pd();
    const __m256i sw = _mm256_loadu_si256((const __m256i *)sw_idx);
    int k = 0;
    for (; k + 2 <= n; k += 2) {
        __m256d va = _mm256_loadu_pd(a + 2 * k);
        __m256d vb = _mm256_loadu_pd(b + 2 * k);
        __m256d t0 = _mm256_mul_pd(va, vb);
        __m256d t1 = _mm256_mul_pd(_mm256_permutevar8x32_pd(sw, va), vb);
        sre = _mm256_add_pd(sre, _mm256_sub_pd(t0, _mm256_permutevar8x32_pd(sw, t0)));
        sim = _mm256_add_pd(sim, _mm256_add_pd(t1, _mm256_permutevar8x32_pd(sw, t1)));
    }
    for (; k < n; k++) {
        double ar = a[2 * k], ai = a[2 * k + 1];
        double br = b[2 * k], bi = b[2 * k + 1];
        *re += ar * br - ai * bi;
        *im += ar * bi + ai * br;
    }
    sre = _mm256_mask_blend_pd(0x5, _mm256_setzero_pd(), sre);
    *re += _mm256_reduce_add_pd(sre);
    sim = _mm256_mask_blend_pd(0x5, _mm256_setzero_pd(), sim);
    *im += _mm256_reduce_add_pd(sim);
}
#else
static inline void dot_cz(const double *restrict a, const double *restrict b, int n,
                          double *re, double *im) {
    double sre = 0.0, sim = 0.0;
    for (int k = 0; k < n; k++) {
        double ar = a[2 * k], ai = a[2 * k + 1];
        double br = b[2 * k], bi = b[2 * k + 1];
        sre += ar * br - ai * bi;
        sim += ar * bi + ai * br;
    }
    *re += sre;
    *im += sim;
}
#endif

void addusxx_g_fp64(const double _Complex *restrict becphi_c, const double _Complex *restrict becpsi_c, const double _Complex *restrict eigts1, const double _Complex *restrict eigts2, const double _Complex *restrict eigts3, const int64_t *restrict ijtoh, const int64_t *restrict ityp, const int64_t *restrict mill, const int64_t *restrict nh_type, const int64_t *restrict nij_type, const int64_t *restrict nl, const int64_t *restrict ofsbeta, const double _Complex *restrict qgm, double _Complex *restrict rhoc, const double *restrict tau, const int64_t *restrict tvanp, const double *restrict xk, const double *restrict xkq, const int64_t nat, const int64_t ngms, const int64_t nhm, const int64_t nij_tot, const int64_t nkb, const int64_t nnr, const int64_t nr1, const int64_t nr2, const int64_t nr3, const int64_t ntyp) {
    (void)nkb;
    (void)nnr;
    if (ngms <= 0 || nat <= 0) return;

    const double *restrict pd = (const double *)becphi_c;
    const double *restrict sd = (const double *)becpsi_c;

    /* eigqts[na] = exp(-i * 2*pi * sum_d (xk[d]-xkq[d]) * tau[d,na]) -- same
     * accumulation order and libm calls as the reference. */
    double *eq = (double *)malloc((size_t)(nat > 0 ? nat : 1) * 2 * sizeof(double));
    const double tpi = 2.0 * 3.141592653589793;
    for (int64_t na = 0; na < nat; ++na) {
        double s0 = (xk[0] - xkq[0]) * tau[0 * nat + na];
        double s1 = (xk[1] - xkq[1]) * tau[1 * nat + na];
        double s2 = (xk[2] - xkq[2]) * tau[2 * nat + na];
        double arg = tpi * ((s0 + s1) + s2); /* same association as the reference */
        double c = cos(arg);
        double sn = sin(arg);
        eq[2 * na] = c;
        eq[2 * na + 1] = -sn;
    }

    /* active atoms in the reference's (nt outer, na inner) order */
    act_t *acts = (act_t *)malloc((size_t)(nat > 0 ? nat : 1) * sizeof(act_t));
    int64_t nact = 0;
    int64_t total_w = 0;
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;
        int64_t nh = nh_type[nt];
        int64_t nbh = nh * (nh + 1) / 2;
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] != nt) continue;
            acts[nact].na = na;
            acts[nact].s = nij_type[nt];
            acts[nact].nbh = (int)nbh;
            acts[nact].eq = eq + 2 * na;
            acts[nact].w = NULL; /* fixed up below */
            nact++;
            total_w += nbh;
        }
    }
    if (nact == 0) { free(eq); free(acts); return; }

    double *w_all = (double *)malloc((size_t)(total_w > 0 ? total_w : 1) * 2 * sizeof(double));
    int64_t wo = 0;
    for (int64_t k = 0; k < nact; ++k) {
        int64_t na = acts[k].na, nt = ityp[na];
        int64_t nh = nh_type[nt];
        int64_t ijkb0 = ofsbeta[na];
        acts[k].w = w_all + 2 * wo;
        for (int64_t ih = 0; ih < nh; ++ih) {
            for (int64_t jh = ih; jh < nh; ++jh) {
                int64_t c = ijtoh[(ih * nhm + jh) * ntyp + nt];
                int64_t ip = 2 * (ijkb0 + ih), jp = 2 * (ijkb0 + jh);
                /* conj(phi_ih)*psi_jh : (pr,-pi)*(sr+si*I) = (pr*sr+pi*si) + (pr*si-pi*sr) I */
                double wr = pd[ip] * sd[jp] + pd[ip + 1] * sd[jp + 1];
                double wi = pd[ip] * sd[jp + 1] - pd[ip + 1] * sd[jp];
                if (jh > ih) {
                    /* + conj(phi_jh)*psi_ih */
                    wr += pd[jp] * sd[ip] + pd[jp + 1] * sd[ip + 1];
                    wi += pd[jp] * sd[ip + 1] - pd[jp + 1] * sd[ip];
                }
                w_all[2 * (wo + c)] = wr;
                w_all[2 * (wo + c) + 1] = wi;
            }
        }
        wo += (int64_t)acts[k].nbh;
    }

    const double *restrict qd = (const double *)qgm;
    const double *restrict e1d = (const double *)eigts1;
    const double *restrict e2d = (const double *)eigts2;
    const double *restrict e3d = (const double *)eigts3;
    double *restrict rd = (double *)rhoc;
    const int64_t two_nij = 2 * nij_tot;

#pragma omp parallel for schedule(static)
    for (int64_t ig = 0; ig < ngms; ++ig) {
        const double *row = qd + ig * two_nij;
        const double *r1 = e1d + (mill[ig] + nr1) * (2 * nat);
        const double *r2 = e2d + (mill[ngms + ig] + nr2) * (2 * nat);
        const double *r3 = e3d + (mill[2 * ngms + ig] + nr3) * (2 * nat);
        int64_t nloc = nl[ig];
        double cre = rd[2 * nloc], cim = rd[2 * nloc + 1];
        for (int64_t k = 0; k < nact; ++k) {
            const act_t *p = &acts[k];
            double dre = 0.0, dim = 0.0;
            dot_cz(p->w, row + 2 * p->s, p->nbh, &dre, &dim);
            int64_t c = 2 * p->na;
            double tr = dre * p->eq[0] - dim * p->eq[1];
            double ti = dre * p->eq[1] + dim * p->eq[0];
            double rr = r1[c], ii = r1[c + 1];
            double tn = tr * rr - ti * ii; ti = tr * ii + ti * rr; tr = tn;
            rr = r2[c]; ii = r2[c + 1];
            tn = tr * rr - ti * ii; ti = tr * ii + ti * rr; tr = tn;
            rr = r3[c]; ii = r3[c + 1];
            tn = tr * rr - ti * ii; ti = tr * ii + ti * rr; tr = tn;
            cre += tr;
            cim += ti;
        }
        rd[2 * nloc] = cre;
        rd[2 * nloc + 1] = cim;
    }

    free(w_all);
    free(acts);
    free(eq);
}
