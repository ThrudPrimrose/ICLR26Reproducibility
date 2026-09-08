#include <stdint.h>
#include <stdio.h>
void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, void *restrict scratch) {
    static int printed = 0;
    if (!printed) {
        fprintf(stderr, "scratch=%p LEN=%ld\n", scratch, (long)LEN_1D);
        fflush(stderr);
        printed = 1;
    }
    const int64_t n = LEN_1D - 1;
    for (int64_t i = 0; i < n; ++i) a[i] = a[i+1] + b[i];
}
