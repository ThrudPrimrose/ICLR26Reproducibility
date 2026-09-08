#include <stdint.h>
#include <omp.h>
#include <stddef.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Compute total number of elements for mapping
    size_t total = (size_t)LEN_2D * (size_t)LEN_2D;

    // Map data to the device once for the whole computation.
    #pragma omp target data map(to: cc[0:total]) map(tofrom: aa[0:total], bb[0:total])
    {
        // Vertical recurrence on aa (independent across columns i)
        #pragma omp target teams distribute parallel for schedule(static) 
        for (int64_t i = 8; i < LEN_2D; ++i) {
            for (int64_t j = 8; j < LEN_2D; ++j) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
            }
        }
        // Horizontal recurrence on bb (independent across rows j)
        #pragma omp target teams distribute parallel for schedule(static) 
        for (int64_t j = 8; j < LEN_2D; ++j) {
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
            }
        }
    }
}
