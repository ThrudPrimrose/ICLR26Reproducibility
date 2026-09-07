/* Optimized 3D heat equation (7-point stencil, A<->B ping-pong).
 *
 * Compared with the naive reference:
 *  - the per-timestep Ac/Bc scratch copies are gone: the interior update is
 *    computed straight from the source array with shifted row pointers
 *    (the reference only needs Ac as a padded copy of the interior, but the
 *    same values are read from A directly),
 *  - the i-slab loop is OpenMP-parallelised (the two half-sweeps stay
 *    sequential in time, as the recurrence requires),
 *  - the inner k loop is auto-vectorised (AVX-512 on the build machine).
 *
 * The floating point expression is written in split statements that mirror
 * the NumPy reference's evaluation order exactly (alpha*(x - 2.0*c + y) with
 * (x - 2.0c) + y, three such terms added left-to-right, plus c), with no
 * FMA-contractible a*b+c patterns, so results stay bit-identical to the
 * NumPy oracle.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline void sweep_slab(const double *restrict src,
                              double *restrict dst,
                              const int64_t N, const double alpha,
                              const int64_t i0, const int64_t i1)
{
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = 1; j < N - 1; ++j) {
            const double *restrict r_ip = src + ((i + 1) * N + j) * N;
            const double *restrict r_im = src + ((i - 1) * N + j) * N;
            const double *restrict r_jp = src + (i * N + j + 1) * N;
            const double *restrict r_jm = src + (i * N + j - 1) * N;
            const double *restrict r_c  = src + (i * N + j) * N;
            double *restrict d          = dst + (i * N + j) * N;
            for (int64_t k = 1; k < N - 1; ++k) {
                double c = r_c[k];
                double u = 2.0 * c;
                double s1 = r_ip[k] - u;
                s1 += r_im[k];
                double s2 = r_jp[k] - u;
                s2 += r_jm[k];
                double s3 = r_c[k + 1] - u;
                s3 += r_c[k - 1];
                double t1 = alpha * s1;
                double t2 = alpha * s2;
                double t3 = alpha * s3;
                double r = t1 + t2;
                r += t3;
                r += c;
                d[k] = r;
            }
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N,
                  const int64_t TSTEPS, const double alpha)
{
    const int64_t ihi = N - 1;
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < ihi; ++i)
            sweep_slab(A, B, N, alpha, i, i + 1);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < ihi; ++i)
            sweep_slab(B, A, N, alpha, i, i + 1);
    }
}
