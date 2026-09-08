#include <stdint.h>
#include <math.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double x = INFINITY;

    #pragma omp parallel for simd reduction(min:x) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        x = (a[i] < x) ? a[i] : x;
    }

    result[0] = x;
}
