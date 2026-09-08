#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    a[0] = b[0] * c[0];

    #pragma omp parallel for simd schedule(static) nontemporal(a) if(LEN_1D > 4096)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
    }
}
