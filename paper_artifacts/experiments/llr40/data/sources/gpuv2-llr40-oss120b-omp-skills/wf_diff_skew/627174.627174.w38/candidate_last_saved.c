#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    // Offload and parallelize the inner loop.
    #pragma omp target parallel map(tofrom: a[0:LEN_2D*LEN_2D])
    {
        for (int64_t i = 1; i < LEN_2D; ++i) {
            double *restrict row_i = a + i * LEN_2D;
            double *restrict row_im1 = a + (i - 1) * LEN_2D;
            #pragma omp for simd schedule(static)
            for (int64_t j = 0; j < LEN_2D - 1; ++j) {
                row_i[j] = row_i[j] + row_im1[j] + row_im1[j + 1];
            }
        }
    }
}
