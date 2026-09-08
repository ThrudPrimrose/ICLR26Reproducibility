#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    printf("LEN_1D=%ld\n", LEN_1D);
    fflush(stdout);
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
