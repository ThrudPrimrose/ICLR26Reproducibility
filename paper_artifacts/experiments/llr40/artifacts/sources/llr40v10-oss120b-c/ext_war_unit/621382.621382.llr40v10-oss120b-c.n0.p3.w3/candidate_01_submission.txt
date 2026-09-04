#include <stdint.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // Assume 64-byte alignment for optimal AVX-512 loads/stores.
    double *restrict a_al = (double *restrict) __builtin_assume_aligned(a, 64);
    const double *restrict b_al = (const double *restrict) __builtin_assume_aligned(b, 64);
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a_al[i] = a_al[i + 1] + b_al[i];
    }
}
