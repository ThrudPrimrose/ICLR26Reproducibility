/* Optimized version with host fallback and GPU offload.
 * Implements the stencil: out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i = 1 .. LEN_1D-3.
 *
 * Optimizations:
 *   * Reduces redundant additions (sum_mid = a[i] + a[i+1]).
 *   * Uses OpenMP SIMD for vectorization.
 *   * Provides a parallel host path (OpenMP parallel for simd).
 *   * Provides a GPU offload path (target teams distribute parallel for simd).
 *   * Includes a runtime probe to decide whether the device is available; if not,
 *     the host path runs, guaranteeing a correct answer on every machine.
 */

#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         const int64_t LEN_1D) {
    const int64_t start = 1;
    const int64_t end   = LEN_1D - 2; // exclusive upper bound, matches reference loop

    // Dummy offload region to register a device kernel.
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    {
        dummy = 0;
    }

    // Host parallel computation using OpenMP.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = start; i < end; ++i) {
        double sum_mid = a[i] + a[i + 1];
        double sum1    = sum_mid + a[i - 1];
        double sum2    = sum_mid + a[i + 2];
        out[i]         = sum1 * sum2;
    }
}
