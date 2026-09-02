#include <stdint.h>
#include <stddef.h>

// Compute out[i] = (a[i]^2 + 1)*(a[i]^2 - 1) = a[i]^4 - 1
void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    #pragma omp parallel for schedule(static) // Parallelize across outer dimension
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        double t = ai * ai;   // a[i]^2
        out[i] = t * t - 1.0; // a[i]^4 - 1
    }
}
