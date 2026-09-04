/* Optimized version of TSVC tsvc_2 microkernel s2233 (fp64).
 * The original kernel computes column-wise prefix sums for `aa` and row-wise
 * prefix sums for `bb` using the operand array `cc`.  This version restructures
 * the two inner loops into separate passes, allowing the compiler to vectorize
 * the inner loop of the `bb` update and enabling OpenMP parallelisation of the
 * outer loop for the `aa` update (which has no cross‑iteration dependencies).
 *
 * The function follows the v2 C‑ABI used by the benchmark harness.
 */

#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
    /* -------------------------------------------------------------------
     * 1) Update `aa` column‑wise.
     *    For each column i (the outer loop) we compute a prefix sum down the
     *    rows j.  The recurrence `aa[j,i] = aa[j‑1,i] + cc[j,i]` carries a
     *    dependence only on the previous row, so different columns are
     *    independent and can be processed in parallel.
     * ------------------------------------------------------------------- */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        for (int64_t j = 8; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    /* -------------------------------------------------------------------
     * 2) Update `bb` row‑wise.
     *    The recurrence `bb[i,j] = bb[i‑1,j] + cc[i,j]` carries a dependence
     *    across rows (i) but each row is independent across columns (j).  The
     *    outer loop must remain sequential, however the inner loop can be
     *    vectorised.  The `omp simd` pragma forces vectorisation without
     *    changing the semantics.
     * ------------------------------------------------------------------- */
    for (int64_t i = 8; i < LEN_2D; ++i) {
        #pragma omp simd
        for (int64_t j = 8; j < LEN_2D; ++j) {
            bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}
