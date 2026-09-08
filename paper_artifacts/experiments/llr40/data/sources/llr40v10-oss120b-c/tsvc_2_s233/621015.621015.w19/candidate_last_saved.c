#include <stdint.h>

/*
 * Optimized implementation of the TSVC "s233" kernel.
 *
 * The kernel computes two independent prefix‑sum style recurrences on two
 * double‑precision matrices `aa` and `bb` using the helper matrix `cc`.
 *
 *   aa[j,i] = aa[j-1,i] + cc[j,i]   (column‑wise recurrence)
 *   bb[j,i] = bb[j,i-1] + cc[j,i]   (row‑wise   recurrence)
 *
 * The original reference implementation used two nested loops with the outer
 * dimension `i`.  That formulation prevents any useful OpenMP parallelism
 * because each outer iteration depends on the previous one for the `bb`
 * recurrence.  We restructure the computation so that the two recurrences are
 * expressed as independent outer loops that can be parallelised safely:
 *   * The column‑wise `aa` computation is independent across columns `i`.
 *   * The row‑wise   `bb` computation is independent across rows    `j`.
 *
 * A single OpenMP parallel region is used with two `omp for` directives; this
 * avoids the overhead of entering and leaving two separate parallel regions.
 *
 * For performance we replace the repeated `j*LEN_2D + i` indexing with pointer
 * arithmetic, reducing the amount of integer multiplication inside the inner
 * loops.  All pointers are marked `restrict` to allow the compiler to assume no
 * aliasing between the three arrays.
 */

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t L = LEN_2D;

    /* Parallel region that contains two independent work‑sharing loops. */
    #pragma omp parallel
    {
        /* -----------------------------------------------------------------
         * Column‑wise prefix sum for `aa`.
         * Each column `i` (starting at index 8) can be processed independently.
         * -------------------------------------------------------------- */
        #pragma omp for schedule(static)
        for (int64_t i = 8; i < L; ++i) {
            /* Start at row 7, the element that the recurrence reads from for
             * the first computed row (j = 8). */
            double *aa_ptr = aa + (7 * L + i);
            const double *cc_ptr = cc + (8 * L + i);

            /* Iterate over rows j = 8 .. L-1.  The stride between consecutive
             * rows in the 1‑D storage is `L`. */
            for (int64_t j = 8; j < L; ++j) {
                aa_ptr += L;               // move to aa[j,i]
                *aa_ptr = *(aa_ptr - L) + *cc_ptr; // aa[j,i] = aa[j-1,i] + cc[j,i]
                cc_ptr += L;               // advance cc pointer to next row
            }
        }

        /* -----------------------------------------------------------------
         * Row‑wise prefix sum for `bb`.
         * Each row `j` (starting at index 8) can be processed independently.
         * -------------------------------------------------------------- */
        #pragma omp for schedule(static)
        for (int64_t j = 8; j < L; ++j) {
            double *bb_ptr = bb + (j * L + 7);   // start at bb[j,7]
            const double *cc_ptr = cc + (j * L + 8); // start at cc[j,8]

            for (int64_t i = 8; i < L; ++i) {
                bb_ptr += 1;               // move to bb[j,i]
                *bb_ptr = *(bb_ptr - 1) + *cc_ptr; // bb[j,i] = bb[j,i-1] + cc[j,i]
                cc_ptr += 1;               // advance cc pointer to next column
            }
        }
    }
}

