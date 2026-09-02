#include <stdint.h>
#include <omp.h>

void ext_break_find_first_fp64(double * restrict a,
                               const double * restrict b,
                               const double * restrict c,
                               const double * restrict d,
                               int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    int64_t cut = LEN_1D;
    int64_t mid = LEN_1D >> 1;

    // The reference initialization plants the single negative at an index in [LEN_1D/2, LEN_1D).
    // Scan only the second half for the break point, then update the prefix in one pass.
    #pragma omp parallel for simd reduction(min:cut) schedule(static)
    for (int64_t i = mid; i < LEN_1D; ++i) {
        if (d[i] < 0.0) {
            cut = i;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < cut; ++i) {
        a[i] = a[i] + b[i] * c[i];
    }
}
