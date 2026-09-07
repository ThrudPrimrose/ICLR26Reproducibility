/* Optimized C implementation of QE us_exx::addusxx_g (flag='c' branch).
 *
 * Reference math (per atom na of species t, per G-vector ig):
 *   aux2 = sum_ih [ sum_jh qgm[ig, nij_t + P(ih,jh)] * bpsi(ih0 + jh) ]
 *            * conj(bphi(ih0 + ih))
 *   rhoc[nl[ig]] += aux2 * eigqts[na] * e1(m1+nr1) * e2(m2+nr2) * e3(m3+nr3)
 * where P = ijtoh[:, :, t] (packed symmetric pair index) and ih0 = ofsbeta[na].
 *
 * Rewritten as a per-species GEMM:
 *   W_n[c] = sum_{(i,j) -> c} bpsi(ih0 + j) * conj(bphi(ih0 + i))   (npair x k)
 *   contrib[ig, n] = dot over c of qgm[ig, nb0_t + c] * W_n[c]
 * The qgm block of a species is streamed once per ig-tile and shared by every
 * atom of that species, and each rhoc cell is updated in L1 by all atoms
 * before leaving the tile.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <tmmintrin.h>
#include <omp.h>

typedef double _Complex cplx;

/* Literal reference (also the safety fallback for out-of-range shapes). */
static void addusxx_literal(
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
    int64_t nr1, int64_t nr2, int64_t nr3, int64_t ntyp)
{
    const double tpi = 2.0 * M_PI;
    cplx *eigqts = (cplx *)malloc((size_t)nat * sizeof(cplx));
    for (int64_t na = 0; na < nat; na++) {
        double arg = 0.0;
        for (int d = 0; d < 3; d++)
            arg += (xk[d] - xkq[d]) * tau[d * nat + na];
        arg *= tpi;
        eigqts[na] = (cplx)((double)cos(arg) - _Complex_I * sin(arg));
    }
    for (int64_t nt = 0; nt < ntyp; nt++) {
        if (!tvanp[nt])
            continue;
        int64_t nij = nij_type[nt];
        for (int64_t na = 0; na < nat; na++) {
            if (ityp[na] != nt)
                continue;
            int64_t ijkb0 = ofsbeta[na];
            for (int64_t ig = 0; ig < ngms; ig++) {
                cplx aux2 = 0.0;
                for (int64_t ih = 0; ih < nh_type[nt]; ih++) {
                    int64_t ikb = ijkb0 + ih;
                    cplx aux1 = 0.0;
                    for (int64_t jh = 0; jh < nh_type[nt]; jh++) {
                        int64_t jkb = ijkb0 + jh;
                        aux1 += qgm[ig * nij_tot + (nij + ijtoh[(ih * nhm + jh) * ntyp + nt])] *
                                becpsi_c[jkb];
                    }
                    aux2 += aux1 * conj(becphi_c[ikb]);
                }
                aux2 = aux2 * eigqts[na] *
                       eigts1[(mill[ig] + nr1) * nat + na] *
                       eigts2[(mill[ngms + ig] + nr2) * nat + na] *
                       eigts3[(mill[2 * ngms + ig] + nr3) * nat + na];
                rhoc[nl[ig]] += aux2;
            }
        }
    }
    free(eigqts);
}

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
    int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    (void)nkb;
    (void)nnr;

    /* Fallback for shapes outside the manifest envelope. */
    int64_t nh_max = 0;
    for (int64_t t = 0; t < ntyp; t++)
        if (nh_type[t] > nh_max)
            nh_max = nh_type[t];
    if (nat > 24 || nh_max > 19 || ntyp > 3) {
        addusxx_literal(becphi_c, becpsi_c, eigts1, eigts2, eigts3, ijtoh, ityp,
                        mill, nh_type, nij_type, nl, ofsbeta, qgm, rhoc, tau, tvanp,
                        xk, xkq, nat, ngms, nhm, nij_tot, nr1, nr2, nr3, ntyp);
        return;
    }

    const double tpi = 2.0 * M_PI;
    const int nthreads = omp_get_max_threads();

    /* ---- per-atom phase factors ---- */
    cplx *eigqts = (cplx *)malloc((size_t)nat * sizeof(cplx));
    for (int64_t na = 0; na < nat; na++) {
        double arg = 0.0;
        for (int d = 0; d < 3; d++)
            arg += (xk[d] - xkq[d]) * tau[d * nat + na];
        arg *= tpi;
        eigqts[na] = (cplx)((double)cos(arg) - _Complex_I * sin(arg));
    }

    /* ---- per-species data ---- */
    int64_t na0[3], kk[3], npair[3], nb0[3];
    cplx *Bp[3];
    int nact = 0;
    cplx *B_all = (cplx *)malloc((size_t)3 * 190 * 24 * sizeof(cplx));
    for (int64_t t = 0; t < ntyp && nact < 3; t++) {
        if (!tvanp[t])
            continue;
        int64_t nh = nh_type[t];
        int64_t np = nh * (nh + 1) / 2;
        int64_t n0 = -1, kc = 0;
        for (int64_t na = 0; na < nat; na++)
            if (ityp[na] == t) {
                if (n0 < 0)
                    n0 = na;
                kc++;
            }
        na0[nact] = n0;
        kk[nact] = kc;
        npair[nact] = np;
        nb0[nact] = nij_type[t];
        Bp[nact] = B_all + (size_t)nact * 190 * 24;
        /* build the weight matrix B[c][jj] */
        cplx *B = Bp[nact];
        memset(B, 0, (size_t)np * kc * sizeof(cplx));
        for (int64_t jj = 0; jj < kc; jj++) {
            int64_t na = n0 + jj;
            int64_t ijkb0 = ofsbeta[na];
            for (int64_t i = 0; i < nh; i++) {
                for (int64_t j = i; j < nh; j++) {
                    int64_t c = ijtoh[(i * nhm + j) * ntyp + t];
                    if (i == j) {
                        B[(size_t)c * kc + jj] += becpsi_c[ijkb0 + i] * conj(becphi_c[ijkb0 + i]);
                    } else {
                        /* both (i,j) and (j,i) map to the same packed column c */
                        B[(size_t)c * kc + jj] += becpsi_c[ijkb0 + j] * conj(becphi_c[ijkb0 + i]) +
                                                  becpsi_c[ijkb0 + i] * conj(becphi_c[ijkb0 + j]);
                    }
                }
            }
        }
        nact++;
    }

    /* ---- tiling of the G-sphere ---- */
    int64_t want = ngms / (4 * nthreads);
    int64_t ntile = want < 32 ? 32 : (want > 1024 ? 1024 : want);
    int64_t ntiles = (ngms + ntile - 1) / ntile;

    /* per-thread R buffer: ntile x nat cplx, zeroed per tile below */
    cplx *R_pool = (cplx *)malloc((size_t)nthreads * (size_t)ntile * (size_t)nat * sizeof(cplx));

    const int64_t *m0 = mill;
    const int64_t *m1 = mill + ngms;
    const int64_t *m2 = mill + 2 * ngms;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        cplx *Rbase = R_pool + (size_t)tid * ntile * nat;

        for (int64_t tile = tid; tile < ntiles; tile += nthr) {
            int64_t ig0 = tile * ntile;
            int64_t nloc = ngms - ig0;
            if (nloc > ntile)
                nloc = ntile;

            for (int s = 0; s < nact; s++) {
                int64_t k = kk[s];
                int64_t np = npair[s];
                int64_t nb = nb0[s];
                int64_t n0 = na0[s];
                cplx *B = Bp[s];
                cplx *R = Rbase + n0; /* nloc x k, row nat */

                /* zero */
                for (int64_t i = 0; i < nloc; i++)
                    for (int64_t jj = 0; jj < k; jj++)
                        R[(size_t)i * nat + jj] = 0.0;

                /* GEMM: R[i][jj] += qgm[(ig0+i)*nij_tot + nb + c] * B[c][jj] */
                for (int64_t c = 0; c < np; c++) {
                    const cplx *Bc = B + (size_t)c * k;
                    const double _Complex *Q = qgm + (ig0 * nij_tot + nb + c);
                    for (int64_t i = 0; i < nloc; i++) {
                        cplx a = Q[i * nij_tot];
                        for (int64_t jj = 0; jj < k; jj++)
                            R[(size_t)i * nat + jj] += a * Bc[jj];
                    }
                }

                /* scale + scatter */
                for (int64_t jj = 0; jj < k; jj++) {
                    int64_t na = n0 + jj;
                    cplx eq = eigqts[na];
                    const double _Complex *e1 = eigts1 + (nr1) * nat + na;
                    const double _Complex *e2 = eigts2 + (nr2) * nat + na;
                    const double _Complex *e3 = eigts3 + (nr3) * nat + na;
                    for (int64_t i = 0; i < nloc; i++) {
                        int64_t ig = ig0 + i;
                        cplx s = eq * e1[m0[ig] * nat] * e2[m1[ig] * nat] * e3[m2[ig] * nat];
                        rhoc[nl[ig]] += R[(size_t)i * nat + jj] * s;
                    }
                }
            }
        }
    }

    free(R_pool);
    free(B_all);
    free(eigqts);
}
