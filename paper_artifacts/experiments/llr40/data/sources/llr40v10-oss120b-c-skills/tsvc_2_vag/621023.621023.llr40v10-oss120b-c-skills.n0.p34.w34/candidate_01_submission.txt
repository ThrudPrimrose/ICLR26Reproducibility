/*
 * Parallel version of the TSVC kernel "vag" for double precision.
 * Copies a[i] = b[ip[i]] using an OpenMP parallel loop.
 *
 * This variant distributes the work across threads to increase memory bandwidth
 * utilization.  No SIMD clause is used to avoid the costly gather simulation
 * observed with the `simd` directive on this kernel.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a,
                     const double *restrict b,
                     const int32_t *restrict ip,
                     const int64_t LEN_1D)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = b[(int64_t)ip[i]];
    }
}
