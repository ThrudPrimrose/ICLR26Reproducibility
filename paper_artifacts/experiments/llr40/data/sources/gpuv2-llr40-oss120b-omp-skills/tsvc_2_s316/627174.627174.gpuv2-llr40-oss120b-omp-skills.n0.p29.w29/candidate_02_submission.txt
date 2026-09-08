#include <stdint.h>
#include <float.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    int on_device = 0;
    #pragma omp target map(from:on_device)
    { on_device = !omp_is_initial_device(); }
    if (on_device) {
        double min_val = DBL_MAX;
        #pragma omp target teams distribute parallel for reduction(min:min_val) map(to:a[0:LEN_1D]) map(from:result[0:1])
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = a[i];
            if (v < min_val) min_val = v;
        }
        result[0] = min_val;
    } else {
        double min_val = DBL_MAX;
        #pragma omp parallel for reduction(min:min_val)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = a[i];
            if (v < min_val) min_val = v;
        }
        result[0] = min_val;
    }
}
