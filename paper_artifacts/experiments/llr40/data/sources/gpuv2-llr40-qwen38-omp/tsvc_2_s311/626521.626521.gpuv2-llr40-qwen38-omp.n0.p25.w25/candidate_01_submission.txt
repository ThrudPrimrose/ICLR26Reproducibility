/* TSVC s311: sum reduction, GPU offload (OpenMP target, explicit memory model). */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out,
                      const int64_t LEN_1D, void *workspace,
                      const int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0) {
        sum_out[0] = 0.0;
        return;
    }
    double s = 0.0;
#pragma omp target map(tofrom: s) map(to: a[0:LEN_1D])
    {
#pragma omp teams distribute parallel for reduction(+:s)
        for (int64_t i = 0; i < LEN_1D; i++) s += a[i];
    }
    sum_out[0] = s;
}
