/* Optimized version of tsvc_2_s275 (TSVC kernel s275).
   Improves memory traffic by keeping the running prefix sum in a register
   (eliminating the load of aa[(j-1)*LEN_2D + i] each iteration).
   Parallelises over columns (i) as they are independent.
*/

#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Each column can be processed independently.
    #pragma omp parallel for schedule(static) default(none) shared(aa,bb,cc,LEN_2D)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double prev = aa[i]; // aa[0, i]
        if (prev > 0.0) {
            // Compute the recurrence a[j,i] = a[j-1,i] + b[j,i] * c[j,i]
            for (int64_t j = 1; j < LEN_2D; ++j) {
                double prod = bb[j * LEN_2D + i] * cc[j * LEN_2D + i];
                prev = prev + prod;
                aa[j * LEN_2D + i] = prev;
            }
        }
    }
}
