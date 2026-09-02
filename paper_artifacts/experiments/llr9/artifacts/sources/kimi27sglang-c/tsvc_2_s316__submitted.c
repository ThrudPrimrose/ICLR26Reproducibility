#include <stddef.h>
#include <stdint.h>

void tsvc_2_s316_fp64(double* restrict a, double* restrict result, int64_t LEN_1D, uint8_t* restrict workspace, int64_t workspace_size)
{
    double x = a[0];
    #pragma omp parallel for reduction(min:x) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        if (a[i] < x) {
            x = a[i];
        }
    }
    result[0] = x;
}
