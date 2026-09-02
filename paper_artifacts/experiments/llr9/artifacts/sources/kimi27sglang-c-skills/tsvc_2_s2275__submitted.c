#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double * restrict a, double * restrict aa,
                       const double * restrict b, const double * restrict bb,
                       const double * restrict c, const double * restrict cc,
                       const double * restrict d, int64_t LEN_2D)
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int64_t j = 0; j < LEN_2D; j++) {
            #pragma omp simd
            for (int64_t i = 0; i < LEN_2D; i++) {
                aa[j * LEN_2D + i] += bb[j * LEN_2D + i] * cc[j * LEN_2D + i];
            }
        }

        #pragma omp for
        for (int64_t i = 0; i < LEN_2D; i++) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}
