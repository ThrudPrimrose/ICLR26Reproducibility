#include <stdint.h>
#include <omp.h>

static __attribute__((noinline)) double sum_serial(const double *restrict a, int64_t n)
{
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i];
    return s;
}

static __attribute__((noinline)) double sum_parallel(const double *restrict a, int64_t n)
{
    double s = 0.0;
#pragma omp parallel for simd reduction(+:s) schedule(static)
    for (int64_t i = 0; i < n; ++i) s += a[i];
    return s;
}

void tsvc_2_s311_fp64(double *restrict a, double *restrict sum_out, int64_t LEN_1D,
                      uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    double s = (LEN_1D < 4096) ? sum_serial(a, LEN_1D) : sum_parallel(a, LEN_1D);
    sum_out[0] = s;
}
