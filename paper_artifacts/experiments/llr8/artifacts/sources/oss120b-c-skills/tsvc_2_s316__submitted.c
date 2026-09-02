#include <stdint.h>
#include <float.h>
#include <omp.h>

void tsvc_2_s316_fp64(double *a, double *result, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    double min_val = DBL_MAX;
    #pragma omp parallel for reduction(min:min_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double val = a[i];
        if (val < min_val) {
            min_val = val;
        }
    }
    result[0] = min_val;
}
