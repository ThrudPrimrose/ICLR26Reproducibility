#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d,
                       const int64_t LEN_2D) {
    // Offload all data to the device, compute, then bring back results.
    #pragma omp target data map(to: b[0:LEN_2D], bb[0:LEN_2D*LEN_2D], c[0:LEN_2D], cc[0:LEN_2D*LEN_2D], d[0:LEN_2D]) \
                            map(tofrom: a[0:LEN_2D], aa[0:LEN_2D*LEN_2D])
    {
        // Update the aa matrix: aa += bb * cc (elementwise)
        #pragma omp target teams distribute parallel for collapse(2)
        for (int64_t j = 0; j < LEN_2D; ++j) {
            for (int64_t i = 0; i < LEN_2D; ++i) {
                int64_t idx = j * LEN_2D + i;
                aa[idx] = aa[idx] + bb[idx] * cc[idx];
            }
        }
        // Update the a vector: a = b + c * d
        #pragma omp target teams distribute parallel for
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}
