#include <stdint.h>
#include <stdio.h>
#include <omp.h>
void scatter_accum_dup_fp64(double *restrict bins, const int32_t *restrict ip,
                              const double *restrict src, const int64_t LEN_1D,
                              uint8_t *restrict workspace, const int64_t workspace_bytes) {
    fprintf(stdout, "LEN=%lld threads=%d\n", (long long)LEN_1D, omp_get_max_threads());
    fflush(stdout);
}
