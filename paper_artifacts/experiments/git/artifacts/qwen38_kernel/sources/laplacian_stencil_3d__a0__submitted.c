#include <stdint.h>
#include <stdlib.h>

/* 8th-order (R=4) central finite-difference 3D Laplacian on a periodic grid,
 * batched over k wavefunctions, fused with per-state kinetic energy.
 *
 * Memory layout: psi/lap are (N, N, N, k) in C row-major, so the flat offset of
 * element (i, j, l, kk) is ((i*N + j)*N + l)*k + kk.  k is the contiguous dim.
 */
void laplacian_stencil_3d_fp64(double *restrict psi, double *restrict lap,
                               double *restrict ekin, int64_t N, int64_t k,
                               double inv_h2) {
    const double C0 = -205.0 / 72.0;
    const double C1 = 8.0 / 5.0;
    const double C2 = -1.0 / 5.0;
    const double C3 = 8.0 / 315.0;
    const double C4 = -1.0 / 560.0;
    const double three_C0 = 3.0 * C0;

#define FOFF(I, J, L) ((((I)*(N) + (J))*(N) + (L)) * (k))

#pragma omp parallel
    {
        double *ek = (double *)malloc(sizeof(double) * (size_t)k);
        for (int64_t kk = 0; kk < k; ++kk) ek[kk] = 0.0;

#pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            int64_t iw[9];
            for (int64_t d = -4; d <= 4; ++d) iw[d + 4] = (i + d + N) % N;
            for (int64_t j = 0; j < N; ++j) {
                int64_t jw[9];
                for (int64_t d = -4; d <= 4; ++d) jw[d + 4] = (j + d + N) % N;
                int64_t ij = ((i * N) + j) * N;
                for (int64_t l = 0; l < N; ++l) {
                    int64_t lw[9];
                    for (int64_t d = -4; d <= 4; ++d) lw[d + 4] = (l + d + N) % N;
                    int64_t o = (ij + l) * k;
                    const int64_t t0 = FOFF(iw[3], j, l), t1 = FOFF(iw[5], j, l);
                    const int64_t t2 = FOFF(iw[2], j, l), t3 = FOFF(iw[6], j, l);
                    const int64_t t4 = FOFF(iw[1], j, l), t5 = FOFF(iw[7], j, l);
                    const int64_t t6 = FOFF(iw[0], j, l), t7 = FOFF(iw[8], j, l);
                    const int64_t u0 = FOFF(i, jw[3], l), u1 = FOFF(i, jw[5], l);
                    const int64_t u2 = FOFF(i, jw[2], l), u3 = FOFF(i, jw[6], l);
                    const int64_t u4 = FOFF(i, jw[1], l), u5 = FOFF(i, jw[7], l);
                    const int64_t u6 = FOFF(i, jw[0], l), u7 = FOFF(i, jw[8], l);
                    const int64_t v0 = FOFF(i, j, lw[3]), v1 = FOFF(i, j, lw[5]);
                    const int64_t v2 = FOFF(i, j, lw[2]), v3 = FOFF(i, j, lw[6]);
                    const int64_t v4 = FOFF(i, j, lw[1]), v5 = FOFF(i, j, lw[7]);
                    const int64_t v6 = FOFF(i, j, lw[0]), v7 = FOFF(i, j, lw[8]);
                    for (int64_t kk = 0; kk < k; ++kk) {
                        double a = three_C0 * psi[o + kk];
                        a += C1 * (psi[t0 + kk] + psi[t1 + kk]);
                        a += C2 * (psi[t2 + kk] + psi[t3 + kk]);
                        a += C3 * (psi[t4 + kk] + psi[t5 + kk]);
                        a += C4 * (psi[t6 + kk] + psi[t7 + kk]);
                        a += C1 * (psi[u0 + kk] + psi[u1 + kk]);
                        a += C2 * (psi[u2 + kk] + psi[u3 + kk]);
                        a += C3 * (psi[u4 + kk] + psi[u5 + kk]);
                        a += C4 * (psi[u6 + kk] + psi[u7 + kk]);
                        a += C1 * (psi[v0 + kk] + psi[v1 + kk]);
                        a += C2 * (psi[v2 + kk] + psi[v3 + kk]);
                        a += C3 * (psi[v4 + kk] + psi[v5 + kk]);
                        a += C4 * (psi[v6 + kk] + psi[v7 + kk]);
                        double lv = inv_h2 * a;
                        lap[o + kk] = lv;
                        ek[kk] += psi[o + kk] * lv;
                    }
                }
            }
        }
#pragma omp critical
        for (int64_t kk = 0; kk < k; ++kk) ekin[kk] += ek[kk];
        free(ek);
    }
    for (int64_t kk = 0; kk < k; ++kk) ekin[kk] = -0.5 * ekin[kk];
}
