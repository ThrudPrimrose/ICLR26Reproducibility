#include <stdint.h>
#include <omp.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    #pragma omp target teams distribute parallel for simd \
        map(to: d[0:LEN_1D], e[0:LEN_1D], c[0:LEN_1D]) \
        map(tofrom: a[0:LEN_1D], b[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        b[i] = d[i] * e[i];
        a[i] += b[i] * c[i];
    }
}
