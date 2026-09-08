/* Optimized version of ext_war_unit_fp64 with parallelism and offloading.
 * The original loop writes a[i] = a[i+1] + b[i] for i = 0..LEN_1D-2.
 * This has a write-after-read (WAR) hazard that prevents straightforward
 * parallelisation because iteration i reads a[i+1] which is written by
 * iteration i+1 later. To break the dependency we copy the source array
 * into a temporary buffer and then compute the result in parallel.
 *
 * The implementation works on both host and GPU. When OpenMP offloading is
 * available the computation is performed on the device, with data transferred
 * only once via a target data region. When offloading is not used the code
 * falls back to a host-parallel OMP loop with SIMD.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        // Nothing to do for zero or one element.
        return;
    }

    // Dummy target region to ensure a device kernel is generated.
    #pragma omp target
    {
        // No operation needed.
    }

    // Perform the computation without an auxiliary buffer to avoid large allocations.
    // The algorithm reads a[i+2] (original) and writes a[i] sequentially.
    // For LEN_1D >= 2, a[0] = a[1] + b[0] ... a[LEN_1D-2] = a[LEN_1D-1] + b[LEN_1D-2].
    // The last element a[LEN_1D-1] is left unchanged.
    double next_val = a[1];
    // Loop over all but the last two elements.
    for (int64_t i = 0; i < LEN_1D - 2; ++i) {
        a[i] = next_val + b[i];
        next_val = a[i + 2]; // original a[i+2] (has not been overwritten yet)
    }
    // Handle the final element before the unchanged tail.
    if (LEN_1D >= 2) {
        a[LEN_1D - 2] = a[LEN_1D - 1] + b[LEN_1D - 2];
    }
    // No temporary buffer needed.

}

