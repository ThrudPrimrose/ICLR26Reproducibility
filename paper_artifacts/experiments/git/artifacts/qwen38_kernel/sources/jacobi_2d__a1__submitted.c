#include <stdint.h>
#include <stddef.h>

/* 5-point Jacobi stencil, two half-sweeps per timestep.
 * Per-element FP order is kept identical to the numpy reference:
 *   0.2 * ((((c + l) + r) + d) + u)
 * so the result is bit-for-bit the reference (up to vector lane independence).
 */
void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS)
{
    const size_t n = (size_t)N;
    const double c2 = 0.2;

    for (int64_t t = 0; t < TSTEPS; ++t) {
        /* B-sweep: read A, write B interior */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            const double *const arow = A + (size_t)i * n;
            const double *const arowU = arow - n;
            const double *const arowD = arow + n;
            double *const brow = B + (size_t)i * n;
            for (int64_t j = 1; j < N - 1; ++j) {
                brow[j] = c2 * (((((arow[j] + arow[j - 1]) + arow[j + 1]) + arowD[j]) + arowU[j]));
            }
        }
        /* A-sweep: read B, write A interior */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < N - 1; ++i) {
            const double *const brow = B + (size_t)i * n;
            const double *const browU = brow - n;
            const double *const browD = brow + n;
            double *const arow = A + (size_t)i * n;
            for (int64_t j = 1; j < N - 1; ++j) {
                arow[j] = c2 * (((((brow[j] + brow[j - 1]) + brow[j + 1]) + browD[j]) + browU[j]));
            }
        }
    }
}
