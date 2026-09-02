/* Heat-3D (PolyBench 3d heat). Optimized in place:
 * - The Ac/Bc "(N-2)^3 scratch copies" of the naive version are pure views of the
 *   center values of A/B (A is not modified while B is computed, and vice versa),
 *   so the stencil reads the center directly: no malloc, no copy passes.
 * - OpenMP threads the 2-D plane loop of each sweep; the inner k-loop is
 *   straight-line and vectorizes at -O3 -march=native.
 * - Per-point arithmetic keeps the reference's exact association:
 *     B = ((alpha*(A[x+]-2.0*c+A[x-]) + alpha*(A[y+]-2.0*c+A[y-]))
 *         + alpha*(A[z+]-2.0*c+A[z-])) + c
 */
#include <stdint.h>
#include <omp.h>

static void sweep(double *restrict dst, const double *restrict src,
                  const int64_t N, const double alpha)
{
    const int64_t NN = N * N;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 1; i < N - 1; ++i)
    for (int64_t j = 1; j < N - 1; ++j) {
        const double *restrict s   = src + i * NN + j * N;
        const double *restrict sxm = s - NN;
        const double *restrict sxp = s + NN;
        const double *restrict sym = s - N;
        const double *restrict syp = s + N;
        double *restrict d         = dst + i * NN + j * N;
        for (int64_t k = 1; k < N - 1; ++k) {
            double c = s[k];
            double t1 = alpha * ((sxp[k] - 2.0 * c) + sxm[k]);
            double t2 = alpha * ((syp[k] - 2.0 * c) + sym[k]);
            double t3 = alpha * ((s[k + 1] - 2.0 * c) + s[k - 1]);
            d[k] = ((t1 + t2) + t3) + c;
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B,
                  const int64_t N, const int64_t TSTEPS, const double alpha)
{
    for (int64_t t = 0; t < TSTEPS; ++t) {
        sweep(B, A, N, alpha);
        sweep(A, B, N, alpha);
    }
}
