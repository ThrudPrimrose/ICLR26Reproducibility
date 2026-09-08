#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    // Offload the element-wise multiplication to the device
    #pragma omp target data map(to: b[0:LEN_1D], c[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
    {
        #pragma omp target teams distribute parallel for
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = b[i] * c[i];
        }
    }
    // Compute inclusive prefix sum on host (sequential, but cheap compared to offloaded work)
    double t = 0.0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double s = a[i];
        a[i] = s + t;
        t = s;
    }
}
