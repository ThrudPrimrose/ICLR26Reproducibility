/* Optimized TSVC s115 kernel */
#include <stddef.h>
#include <omp.h>
#include <stdint.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_2D <= 1) return;
    #pragma omp parallel
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double aj = a[j];
        int64_t row = j * LEN_2D;
        int64_t i_start = j + 1;
        int64_t count = LEN_2D - i_start;
        double *ap = a + i_start;
        const double *aarr = aa + row + i_start;
        #pragma omp for simd aligned(ap,aarr:64) schedule(static)
        for (int64_t k = 0; k < count; ++k) {
            ap[k] -= aarr[k] * aj;
        }
    }
}
