/* Optimized version of TSVC tsvc_2_s2710 microkernel.
   Original reference implementation in /shared/tasks/tsvc_2_s2710/tsvc_2_s2710_reference.c.
   This version moves loop-invariant conditions out of the loop,
   adds OpenMP parallelism with SIMD vectorization, and uses restrict pointers.
*/

#include <stdint.h>
#include <stdbool.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const bool large = LEN_1D > 10;        // loop‑invariant condition
    const bool pos   = x[0] > 0.0;         // loop‑invariant condition on leading scalar

    // Parallelize across threads and let the compiler SIMD‑vectorize the innermost loop.
    // "simd" clause encourages vectorization despite the conditional statements.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        double bi = b[i];
        double di = d[i];
        double ei = e[i];
        double ci = c[i];
        bool cond = ai > bi;               // data‑dependent branch
        if (cond) {
            // a[i] gets updated, b[i] unchanged
            a[i] = ai + bi * di;
            if (large) {
                // c[i] += d[i] * d[i]
                c[i] = ci + di * di;
            } else {
                // c[i] = d[i] * e[i] + 1.0
                c[i] = di * ei + 1.0;
            }
        } else {
            // b[i] gets updated, a[i] unchanged
            b[i] = ai + ei * ei;
            if (pos) {
                // c[i] = a[i] + d[i] * d[i]
                c[i] = ai + di * di;
            } else {
                // c[i] += e[i] * e[i]
                c[i] = ci + ei * ei;
            }
        }
    }
}

