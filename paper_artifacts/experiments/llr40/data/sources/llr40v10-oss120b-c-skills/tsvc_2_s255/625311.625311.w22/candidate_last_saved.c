/* Optimized version of TSVC tsvc_2_s255 kernel.
 * Rewrites the original recurrence into an explicit 3-point stencil with wrap-around.
 * This removes loop-carried dependencies, enabling parallelism and vectorization.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // Guard against zero length – nothing to do.
    // Guard against zero or one-length arrays – the reference kernel expects LEN_1D >= 2.
    if (LEN_1D <= 0) {
        return;
    }
    const double factor = 0.333; // same constant as reference implementation
    if (LEN_1D == 1) {
        // With only one element, the original recurrence would read out‑of‑bounds.
        // Define the result as the average of three copies of b[0] for safety.
        a[0] = (b[0] + b[0] + b[0]) * factor;
        return;
    }
    // Handle the first two elements separately to account for wrap‑around.
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * factor;
    if (LEN_1D >= 2) {
        a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * factor;
    }

    // The main loop has no cross‑iteration dependence and uses unit‑stride accesses.
    // Parallelize across outer iterations; SIMD is left to the compiler.
    #pragma omp parallel for schedule(static) // no reduction needed
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * factor;
    }
}

