#include <stdint.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Parallelize across outer loop; each iteration is independent.
    // Combine producer and consumer loops into a single fused loop and
    // let the compiler vectorize with SIMD via the combined parallel for simd directive.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double t = a[i];        // Load a[i]
        t *= t;                 // t = a[i]^2
        out[i] = t * t - 1.0;   // out[i] = a[i]^4 - 1
    }
}
