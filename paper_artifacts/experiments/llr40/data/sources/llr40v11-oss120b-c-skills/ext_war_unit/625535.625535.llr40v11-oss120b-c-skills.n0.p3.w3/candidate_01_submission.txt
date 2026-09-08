#include <stdint.h>
#include <string.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return;
    }

    // Shift a left by one element safely with memmove (handles overlap)
    memmove(a, a + 1, (size_t)(LEN_1D - 1) * sizeof(double));

    // Add b to the shifted values in parallel
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = a[i] + b[i];
    }
}
