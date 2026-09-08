/* Optimized version of tsvc_2_s255 kernel.
 * Computes a[i] = (b[i] + b[i-1] + b[i-2]) / 3 with wrap-around.
 * The original reference used a recurrence to provide the previous two b values.
 * By expressing the recurrence analytically we remove loop-carried dependencies,
 * enabling parallelism and vectorization.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const double inv3 = 0.333;
    if (LEN_1D == 1) {
        // All indices wrap to the sole element.
        a[0] = (b[0] + b[0] + b[0]) * inv3;
        return;
    }
    // Pre‑compute the wrap‑around values for the first two positions.
    const double b_last   = b[LEN_1D - 1];   // b[-1]
    const double b_last2  = b[LEN_1D - 2];   // b[-2]

    // i = 0
    a[0] = (b[0] + b_last + b_last2) * inv3;
    // i = 1
    a[1] = (b[1] + b[0] + b_last) * inv3;

    // Main body: i >= 2, no wrap‑around needed.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i-1] + b[i-2]) * inv3;
    }
}

