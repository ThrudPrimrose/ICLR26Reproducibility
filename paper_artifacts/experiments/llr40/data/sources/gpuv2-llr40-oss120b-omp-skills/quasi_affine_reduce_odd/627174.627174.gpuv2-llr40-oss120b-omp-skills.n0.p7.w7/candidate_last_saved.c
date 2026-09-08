#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Number of odd indices is LEN_1D / 2 (integer division)
    int64_t n = LEN_1D / 2;
    double dummy = 0.0;
    #pragma omp target map(tofrom: dummy)
    {
        dummy = 0.0;
    }
    double acc = 0.0;
    #pragma omp parallel for simd reduction(+:acc) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        acc += a[2 * i + 1];
    }
    out[0] = acc;
}
