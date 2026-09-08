#include <stdint.h>
#include <omp.h>
#include <float.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double x = DBL_MAX;
    #pragma omp parallel for reduction(min:x) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] < x) {
            x = a[i];
        }
    }
    result[0] = x;
}
