#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(double *a, double *out, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    #pragma omp parallel for schedule(static) default(none) shared(a, out, LEN_1D)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        double t = ai * ai;
        double u = t + 1.0;
        double v = t - 1.0;
        out[i] = u * v;
    }
}
