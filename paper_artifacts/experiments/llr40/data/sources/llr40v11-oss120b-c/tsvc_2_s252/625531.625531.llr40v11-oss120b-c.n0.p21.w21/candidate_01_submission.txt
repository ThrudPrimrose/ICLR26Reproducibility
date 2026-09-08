/* Optimized version of tsvc_2_s252_fp64 kernel.
   Computes a[i] = b[i] * c[i] + (i>0 ? b[i-1] * c[i-1] : 0).
   This eliminates the loop-carried dependency, enabling vectorization and
   parallelization.
*/

#include <stdint.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // Compute first element separately (no predecessor).
    a[0] = b[0] * c[0];
    if (LEN_1D == 1) return;
    // Parallelize the remaining iterations.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double prod = b[i] * c[i];
        double prev = b[i-1] * c[i-1];
        a[i] = prod + prev;
    }
}
