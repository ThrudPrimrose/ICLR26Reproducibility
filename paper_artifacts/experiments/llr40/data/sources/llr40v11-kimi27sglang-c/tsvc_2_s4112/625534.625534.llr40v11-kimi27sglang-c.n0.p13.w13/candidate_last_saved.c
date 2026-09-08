#include <stdint.h>
#include <stdio.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    static int count = 0;
    if (count < 5) {
        fprintf(stderr, "LEN_1D=%ld\n", (long)LEN_1D);
        fflush(stderr);
        count++;
    }
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
