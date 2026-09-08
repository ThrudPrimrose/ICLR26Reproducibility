#include <stdint.h>
#include <math.h>
#include <stdio.h>

static int g_diag = 0;

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    if (!g_diag) {
        g_diag = 1;
        printf("DIAG LEN_1D=%ld\n", (long)LEN_1D);
        fflush(stdout);
    }
    if (LEN_1D <= 0) { result[0] = 0.0; return; }

    double m;
    #pragma omp target teams distribute parallel for \
        map(to: a[0:LEN_1D]) map(from: result[0:1]) reduction(min: m)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] < m) m = a[i];
    }
    result[0] = m;
}
