#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 3) return;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n - 2; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
}
