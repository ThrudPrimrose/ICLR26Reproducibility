#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t n = LEN_1D / 2;
    double acc = 0.0;

    if (n <= 4096) {
        #pragma omp simd reduction(+:acc)
        for (int64_t j = 0; j < n; ++j) {
            acc += a[1 + 2 * j];
        }
    } else {
        #pragma omp parallel for simd reduction(+:acc) schedule(static)
        for (int64_t j = 0; j < n; ++j) {
            acc += a[1 + 2 * j];
        }
    }

    out[0] = acc;
}
