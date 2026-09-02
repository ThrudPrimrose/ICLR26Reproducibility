/*
 * TSVC_2 kernel "s2710" implementation in C.
 *
 * The reference Python implementation (found in /shared/tasks/tsvc_2_s2710/tsvc_2_s2710_numpy.py)
 * defines the following computation for each index i in [0, LEN_1D):
 *
 *   if (a[i] > b[i]) {
 *       a[i] = a[i] + b[i] * d[i];
 *       if (LEN_1D > 10) {
 *           c[i] = c[i] + d[i] * d[i];
 *       } else {
 *           c[i] = d[i] * e[i] + 1.0;
 *       }
 *   } else {
 *       b[i] = a[i] + e[i] * e[i];
 *       if (x[0] > 0.0) {
 *           c[i] = a[i] + d[i] * d[i];
 *       } else {
 *           c[i] = c[i] + e[i] * e[i];
 *       }
 *   }
 *
 * The kernel has no cross‑iteration dependencies: each iteration reads and writes only
 * the elements at index i.  Therefore the loop can be safely parallelised with OpenMP
 * and vectorised with SIMD.  The two loop‑invariant conditions (`LEN_1D > 10` and
 * `x[0] > 0.0`) are hoisted out of the loop to enable the compiler to generate a tight
 * vectorised inner loop.
 *
 * The function follows the exact signature expected by the benchmark harness:
 *
 *   void tsvc_2_s2710_fp64(double *restrict a,
 *              double *restrict b,
 *              double *restrict c,
 *              const double *restrict d,
 *              const double *restrict e,
 *              const double *restrict x,
 *              int64_t LEN_1D, uint8_t *workspace, int64_t workspace_len);
 *
 * All pointer arguments are marked `restrict` to give the compiler the alias‑free guarantee
 * required for maximal vectorisation.  The input‑only pointers (`d`, `e`, `x`) are also `const`.
 *
 * The implementation uses a combined `#pragma omp parallel for simd` construct with a static
 * schedule, which gives each thread a contiguous chunk of the iteration space — ideal for the
 * row‑major layout of 1‑D arrays and avoids false sharing on the output arrays.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a,
           double *restrict b,
           double *restrict c,
           const double *restrict d,
           const double *restrict e,
           const double *restrict x,
           int64_t LEN_1D, uint8_t *workspace, int64_t workspace_len)
{
    const int64_t len = LEN_1D;
    const int64_t len_gt10 = (len > 10);
    const double x0_val = x[0];
    const int64_t x0_gt0 = (x0_val > 0.0);

    /* Parallelise the outer loop. The loop body contains only simple arithmetic and
       data‑independent branches, allowing the compiler to emit SIMD instructions. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < len; ++i) {
        if (a[i] > b[i]) {
            a[i] = a[i] + b[i] * d[i];
            if (len_gt10) {
                c[i] = c[i] + d[i] * d[i];
            } else {
                c[i] = d[i] * e[i] + 1.0;
            }
        } else {
            b[i] = a[i] + e[i] * e[i];
            if (x0_gt0) {
                c[i] = a[i] + d[i] * d[i];
            } else {
                c[i] = c[i] + e[i] * e[i];
            }
        }
    }
}

