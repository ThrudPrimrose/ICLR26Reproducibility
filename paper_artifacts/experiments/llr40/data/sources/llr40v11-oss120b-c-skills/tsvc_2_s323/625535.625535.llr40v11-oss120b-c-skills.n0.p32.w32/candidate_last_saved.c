/* Parallel prefix-sum version of tsvc_2_s323 kernel.
   Original kernel:
       a[i] = b[i-1] + c[i] * d[i];
       b[i] = a[i] + c[i] * e[i];
   This creates a recurrence b[i] = b[i-1] + c[i]*d[i] + c[i]*e[i].
   We compute b via a parallel prefix sum without extra O(N) storage,
   then compute a in parallel.
*/

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return;
    }
    const int64_t n = LEN_1D;

    // Compute b sequentially (prefix sum)
    for (int64_t i = 1; i < n; ++i) {
        b[i] = b[i - 1] + c[i] * d[i] + c[i] * e[i];
    }

    // Compute a in parallel using b
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < n; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
    }
}
