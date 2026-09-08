/* ext_break_capture – find the first element greater than a constant threshold.
 *
 * This implementation follows the reference semantics:
 *   * The output arrays (out_index, out_value) have a single element each.
 *   * They are pre-initialized to sentinel values (-1 and -1.0) by the caller, but we set them
 *     explicitly for correctness.
 *   * The threshold is the constant 1.0 (the reference kernel uses `k = 1`).
 *
 * The original reference walks the array sequentially and breaks on the first match. That is
 * fast when the match occurs early, but for the benchmark the crossing index scales with the
 * array size (≈40‑70% of the length). A parallel reduction that scans the whole array can
 * therefore achieve a speed‑up on large inputs.
 *
 * We use an OpenMP `parallel for` with a `reduction(min: min_idx)` to compute the smallest
 * index where `a[i] > k`. The reduction gives each thread a private copy of `min_idx`
 * initialised to the maximum value of the type; we then keep the smallest matching index per
 * thread with an explicit `if (i < min_idx)` guard. After the parallel region we test whether a
 * match was found and write the output values.
 *
 * The loop is vectorised by the compiler (the predicate is a simple compare) and the OpenMP
 * runtime distributes the iterations across all cores. No `aligned` clauses are used – the input
 * pointer `a` has only its natural alignment, which is sufficient for safe vectorisation.
 */

#include <stdint.h>
#include <omp.h>

/* The ABI expects the name `ext_break_capture_fp64`. The function receives a read‑only array
 * `a`, output scalars `out_index` and `out_value`, and the length of the 1‑D input.
 */
void ext_break_capture_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0; // threshold
    // Initialise outputs to sentinel values; required by the spec.
    out_index[0] = -1;
    out_value[0] = -1.0;

    // Sentinel for "not found" – any real index is < LEN_1D.
    int64_t min_idx = LEN_1D;

    #pragma omp parallel for reduction(min:min_idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > k) {
            if (i < min_idx) {
                min_idx = i;
            }
        }
    }

    if (min_idx < LEN_1D) {
        out_index[0] = min_idx;
        out_value[0] = a[min_idx];
    }
    // else leave sentinel values.
    return;
}
