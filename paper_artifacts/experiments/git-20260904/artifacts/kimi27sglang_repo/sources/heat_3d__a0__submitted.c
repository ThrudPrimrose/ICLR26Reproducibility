#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#ifndef HEAT_BI
#define HEAT_BI 32
#endif
#ifndef HEAT_BJ
#define HEAT_BJ 32
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline void half_step(const double *restrict src,
                             double *restrict dst,
                             int64_t n,
                             double alpha)
{
    const int64_t nn = n * n;
    const int64_t Bi = HEAT_BI;
    const int64_t Bj = HEAT_BJ;

    #pragma omp for collapse(2) schedule(static)
    for (int64_t i0 = 1; i0 < n - 1; i0 += Bi) {
        for (int64_t j0 = 1; j0 < n - 1; j0 += Bj) {
            const int64_t imax = MIN(i0 + Bi, n - 1);
            const int64_t jmax = MIN(j0 + Bj, n - 1);
            for (int64_t i = i0; i < imax; ++i) {
                for (int64_t j = j0; j < jmax; ++j) {
                    const int64_t base = ((i * n) + j) * n;
                    #pragma omp simd safelen(8)
                    for (int64_t k = 1; k < n - 1; ++k) {
                        const int64_t idx = base + k;
                        const double c = src[idx];
                        const double dx = src[idx + nn] - 2.0 * c + src[idx - nn];
                        const double dy = src[idx + n]  - 2.0 * c + src[idx - n];
                        const double dz = src[idx + 1]   - 2.0 * c + src[idx - 1];
                        dst[idx] = alpha * dx + alpha * dy + alpha * dz + c;
                    }
                }
            }
        }
    }
}

void heat_3d_fp64(double *restrict A,
                  double *restrict B,
                  const int64_t N,
                  const int64_t TSTEPS,
                  const double alpha)
{
    #pragma omp parallel
    {
        for (int64_t t = 1; t < TSTEPS + 1; ++t) {
            half_step(A, B, N, alpha);
            half_step(B, A, N, alpha);
        }
    }
}
