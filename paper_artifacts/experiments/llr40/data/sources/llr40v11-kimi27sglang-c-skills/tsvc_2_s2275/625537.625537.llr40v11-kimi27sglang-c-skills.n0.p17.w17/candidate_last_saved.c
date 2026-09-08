#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
    const int64_t n = LEN_2D;

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int64_t j = 0; j < n; ++j) {
            #pragma omp simd nontemporal(aa)
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx = j * n + i;
                aa[idx] = aa[idx] + bb[idx] * cc[idx];
            }
        }

        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}
