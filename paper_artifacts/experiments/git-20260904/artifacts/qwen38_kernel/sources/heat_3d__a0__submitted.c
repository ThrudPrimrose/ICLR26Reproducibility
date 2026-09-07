#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* 3-D explicit heat diffusion, 7-point stencil, in-place two-sweep per timestep.
 * A and B are N x N x N C-contiguous; only the interior [1,N-2]^3 is updated. */
void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N64, int64_t T64, double alpha) {
    const int N = (int)N64;
    const int T = (int)T64;
    if (N < 3 || T <= 0) return;
    const int Nn = N - 2;
    const int64_t N2 = (int64_t)N * (int64_t)N;
    for (int t = 0; t < T; ++t) {
        /* sweep A -> B */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                const double *restrict c  = A + N2 * i + (int64_t)N * j + 1;
                const double *restrict xp = c + N2;
                const double *restrict xm = c - N2;
                const double *restrict yp = c + N;
                const double *restrict ym = c - N;
                double *restrict b = B + N2 * i + (int64_t)N * j + 1;
                for (int k = 0; k < Nn; ++k) {
                    b[k] = alpha * (xp[k] - 2.0 * c[k] + xm[k])
                         + alpha * (yp[k] - 2.0 * c[k] + ym[k])
                         + alpha * (c[k + 1] - 2.0 * c[k] + c[k - 1])
                         + c[k];
                }
            }
        }
        /* sweep B -> A */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                const double *restrict c  = B + N2 * i + (int64_t)N * j + 1;
                const double *restrict xp = c + N2;
                const double *restrict xm = c - N2;
                const double *restrict yp = c + N;
                const double *restrict ym = c - N;
                double *restrict b = A + N2 * i + (int64_t)N * j + 1;
                for (int k = 0; k < Nn; ++k) {
                    b[k] = alpha * (xp[k] - 2.0 * c[k] + xm[k])
                         + alpha * (yp[k] - 2.0 * c[k] + ym[k])
                         + alpha * (c[k + 1] - 2.0 * c[k] + c[k - 1])
                         + c[k];
                }
            }
        }
    }
}
