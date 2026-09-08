#include <stdint.h>
#include <float.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double min_val = DBL_MAX;
    #pragma omp parallel for simd reduction(min:min_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] < min_val) {
            min_val = a[i];
        }
    }
    result[0] = min_val;
}
