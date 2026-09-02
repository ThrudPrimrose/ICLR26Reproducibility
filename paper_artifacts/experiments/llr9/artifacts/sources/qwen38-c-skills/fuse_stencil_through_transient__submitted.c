#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         int64_t LEN_1D,
                                         uint8_t *workspace,
                                         int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D < 4) return;
    int64_t hi = LEN_1D - 2; /* i = 1 .. hi-1 */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < hi; ++i) {
        double s0 = a[i - 1] + a[i] + a[i + 1];
        double s1 = a[i] + a[i + 1] + a[i + 2];
        out[i] = s0 * s1;
    }
}
