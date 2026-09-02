/*
 * ext_break_find_first - find-first break pattern (TSVC s481)
 *
 * The reference NumPy implementation (Python) is:
 *   for i in range(LEN_1D):
 *       if d[i] < 0.0:
 *           break
 *       a[i] = a[i] + b[i] * c[i]
 *
 * The benchmark harness passes four 1-D contiguous arrays of the same length and an
 * integer length. All arrays are distinct (the reference creates them separately) and are
 * double-precision (float64). The kernel must update a in-place and perform the arithmetic only
 * up to (but not including) the first negative entry in d.
 *
 * Strategy
 * --------
 * * First locate the break index with a simple scalar scan. This scan is cheap - it executes at
 *   most LEN_1D iterations and exits as soon as the negative element is found.
 * * Once the index is known we can run the remaining arithmetic as a separate loop that the
 *   compiler can fully vectorise and, optionally, parallelise with OpenMP. Because the break
 *   index is known beforehand there is no data-dependent branch inside the compute loop, allowing
 *   the optimiser to emit SIMD instructions.
 * * The implementation uses restrict qualifiers (the benchmark guarantees that the four
 *   arrays do not alias) and a parallel for with static scheduling - a cheap way to get
 *   multi-core speed-up when the break occurs far into the array. If the break occurs very early
 *   the overhead of the OpenMP team creation is negligible compared with the baseline serial
 *   version.
 *
 * The function name and signature follow the benchmark ABI: the kernel name is exactly the
 * task key (the harness looks for a function with this name) and the arguments are pointers to the
 * data followed by the length. All pointers are to double and the length is an int64_t.
 * The symbol name must be ext_break_find_first_fp64 and include workspace arguments (ignored here).
 */

#include <omp.h>
#include <stdint.h>

/*
 * Kernel entry point.
 *   a  - input/output array (modified in place)
 *   b  - input array
 *   c  - input array
 *   d  - input array containing a single negative element defining the break point
 *   LEN_1D - number of elements in each array
 *   workspace - ignored pointer
 *   workspace_bytes - ignored size
 */
void ext_break_find_first_fp64(double * restrict a,
                         const double * restrict b,
                         const double * restrict c,
                         const double * restrict d,
                         int64_t LEN_1D,
                         uint8_t * restrict workspace,
                         int64_t workspace_bytes)
{
    // Avoid unused parameter warnings
    (void)workspace;
    (void)workspace_bytes;

    int64_t break_idx = LEN_1D; // default: no break found (should not happen in the benchmark)

    // Find the first negative value in d
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (d[i] < 0.0) {
            break_idx = i;
            break;
        }
    }

    // Compute the expression up to break_idx. The loop can be vectorised and, for large
    // break_idx, parallelised across threads.
    #pragma omp parallel for schedule(static) if(break_idx > 0)
    for (int64_t i = 0; i < break_idx; ++i) {
        a[i] = a[i] + b[i] * c[i];
    }
}
