/* Parallel column-wise version of TSVC tsvc_2 s2233 microkernel.
 *
 * The original kernel performs two vertical prefix-sum operations:
 *   1) aa[j,i] = aa[j-1,i] + cc[j,i]
 *   2) bb[i,j] = bb[i-1,j] + cc[i,j]
 *
 * Both updates are independent per column, so we parallelise across columns
 * (the outer index).  Within each column we keep a scalar accumulator to
 * avoid repeatedly loading the previous element.  This preserves the exact
 * recurrence while enabling OpenMP threading.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa,
                       double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D) {
    const int64_t start = 8; // lower bound for both dimensions
    #pragma omp parallel for schedule(static)
    for (int64_t i = start; i < LEN_2D; ++i) {
        // ---- Prefix sum for column i of aa ----
        double acc_aa = aa[(start - 1) * LEN_2D + i]; // aa[7,i]
        for (int64_t j = start; j < LEN_2D; ++j) {
            acc_aa += cc[j * LEN_2D + i];
            aa[j * LEN_2D + i] = acc_aa;
        }

        // ---- Prefix sum for column i of bb ----
        double acc_bb = bb[(start - 1) * LEN_2D + i]; // bb[7,i]
        for (int64_t j = start; j < LEN_2D; ++j) {
            acc_bb += cc[j * LEN_2D + i];
            bb[j * LEN_2D + i] = acc_bb;
        }
    }
}

