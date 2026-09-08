#include <stdint.h>
#include <stddef.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    // Handle empty case
    if (LEN_1D <= 0) return;
    // First element has no predecessor
    a[0] = b[0] * c[0];
    // Process remaining elements; each a[i] = b[i]*c[i] + b[i-1]*c[i-1]
    // This loop has no data dependence on a, so it can be parallelized and vectorized.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i] * c[i] + b[i-1] * c[i-1];
    }
}
