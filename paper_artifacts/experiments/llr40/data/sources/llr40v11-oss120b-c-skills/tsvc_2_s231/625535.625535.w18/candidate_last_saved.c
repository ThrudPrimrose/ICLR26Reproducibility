#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Vectorized column-major traversal.
    // Outer loop over rows (j) respects the recurrence across rows.
    // Inner loop over columns (i) is independent and can be SIMD-vectorized.
    // Using contiguous memory access for the inner loop yields high performance.
    for (int64_t j = 1; j < LEN_2D; ++j) {
        #pragma omp simd aligned(aa,bb:64)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
        }
    }
}
