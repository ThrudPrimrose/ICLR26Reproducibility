#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, int64_t n,
                      uint8_t *workspace, int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        double ai = a[i] + b[i] * c[i];
        a[i] = ai;
        double prev = aa[i];
        for (int64_t j = 1; j < n; j++)
            aa[j * n + i] = prev = prev + bb[j * n + i] * ai;
    }
}
