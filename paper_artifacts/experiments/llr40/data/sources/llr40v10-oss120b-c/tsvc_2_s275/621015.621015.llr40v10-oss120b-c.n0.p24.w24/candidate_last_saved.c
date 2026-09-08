/* Optimized implementation of the TSVC tsvc_2_s275 kernel.
   The reference computes, for each column i, a recurrence over rows j:
     aa[j,i] = aa[j-1,i] + bb[j,i] * cc[j,i]
   if the initial element aa[0,i] > 0.
   This version processes rows in the outer loop and columns in blocks,
   keeping the per‑column running sum in a register. It parallelises the
   block loop with OpenMP and accesses the arrays row‑major, which gives
   contiguous memory accesses and allows the compiler to vectorise the
   inner column loop.
*/

#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t B = 8; // block size (must divide dimensions for best SIMD)
    #pragma omp parallel for schedule(static)
    for (int64_t ib = 0; ib < LEN_2D; ib += B) {
        int64_t block = B;
        if (ib + block > LEN_2D) {
            block = LEN_2D - ib; // handle tail
        }
        /* per‑column state for this block */
        double prev[8];
        char   flag[8];
        /* initialise running sum and active flag for each column */
        for (int64_t k = 0; k < block; ++k) {
            int64_t i = ib + k;
            double a0 = aa[i];
            flag[k] = (a0 > 0.0);
            prev[k] = a0;
        }
        /* scan down the rows */
        for (int64_t j = 1; j < LEN_2D; ++j) {
            int64_t base = j * LEN_2D + ib;          // start index of this row for the block
            for (int64_t k = 0; k < block; ++k) {
                if (flag[k]) {
                    double prod = bb[base + k] * cc[base + k];
                    double cur  = prev[k] + prod;
                    aa[base + k] = cur;
                    prev[k] = cur;
                }
            }
        }
    }
}

