#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    printf("PROBE LEN_2D=%lld max_threads=%d\n", (long long)LEN_2D, omp_get_max_threads());
    fflush(stdout);
    int x = 0;
    #pragma omp target map(tofrom: x)
    x = 1;
    (void)aa; (void)bb; (void)cc;
    if (x < 0) aa[0] = 1.0;
}
