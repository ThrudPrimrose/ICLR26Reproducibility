/* Optimized implementation of the ``segment_reduce_ragged`` kernel.
 *
 * The reference implementation (generated automatically from the NumPy
 * reference) computes a dot-product for each ragged segment. The outer
 * loop iterates over ``NSEG`` segments, the inner loop reduces the product of
 * ``val`` and ``w`` over the range ``[row_ptr[s], row_ptr[s+1])``.
 *
 * This version adds OpenMP parallelisation of the outer loop with a dynamic
 * schedule to balance work across segments of highly irregular length. The
 * inner reduction is additionally marked with ``#pragma omp simd`` so the
 * compiler can emit SIMD instructions while preserving the floating-point
 * reduction semantics via a reduction clause. All pointers are ``restrict``
 * (the ABI requires it) which allows the compiler to assume no aliasing.
 */

#include <stdint.h>
#include <omp.h>

/* The kernel name follows the canonical C‑ABI naming convention used by the
 * harness: ``<kernel>_fp64`` for double‑precision inputs. */
void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG) {
    /* Parallelise over segments. ``schedule(dynamic)`` is chosen because the
     * segment lengths follow a log‑normal distribution and can vary widely – a
     * static partition would lead to severe load imbalance. */
        for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        const int64_t start = row_ptr[s];
        const int64_t end   = row_ptr[s + 1];
        /* SIMD‑vectorise the inner product while safely reducing into
         * ``acc``. The reduction clause guarantees thread‑local accumulation
         * and permits the compiler to emit vector instructions. */
        #pragma omp simd reduction(+:acc)
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
