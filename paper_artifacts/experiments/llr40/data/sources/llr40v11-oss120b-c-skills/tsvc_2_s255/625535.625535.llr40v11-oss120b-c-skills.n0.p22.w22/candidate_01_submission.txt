/* Optimized version of TSVC tsvc_2 s255 kernel.
 * Original reference computes a[i] = (b[i] + x + y) * 0.333 where
 * x and y are a shift register of the last two b values, wrapping at the end.
 * The recurrence can be expressed directly with a circular index, removing the
 * loop-carried dependence and exposing parallelism and vectorization.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const double factor = 0.333;
    if (LEN_1D <= 0) return;
    // handle the first two elements explicitly (wrap‑around)
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * factor;
    if (LEN_1D == 1) return; // nothing more to do
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * factor;
    // main loop – now independent, suitable for OpenMP parallelism and SIMD
    #pragma omp parallel for schedule(static)
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * factor;
    }
}
