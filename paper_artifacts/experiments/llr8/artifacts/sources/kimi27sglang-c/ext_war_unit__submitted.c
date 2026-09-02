#include <stdint.h>

void ext_war_unit_fp64(double *restrict a, double *restrict b, int64_t LEN_1D,
                       uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t n = LEN_1D - 1;
    double *aa = __builtin_assume_aligned(a, 32);
    double *bb = __builtin_assume_aligned(b, 32);
    for (int64_t i = 0; i < n; ++i) {
        aa[i] = aa[i + 1] + bb[i];
    }
}
