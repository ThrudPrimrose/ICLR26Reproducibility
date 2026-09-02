#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(double *a, double *out, int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const double *__restrict av = a;
    double *__restrict o = out;

    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double t = av[i] * av[i];
        o[i] = (t + 1.0) * (t - 1.0);
    }
}
