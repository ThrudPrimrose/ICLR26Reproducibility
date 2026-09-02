#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    double acc = 0.0;
#pragma omp parallel for reduction(+:acc) schedule(static)
    for (int64_t i = 1; i < LEN_1D; i += 2) {
        acc += a[i];
    }
    out[0] = acc;
}
