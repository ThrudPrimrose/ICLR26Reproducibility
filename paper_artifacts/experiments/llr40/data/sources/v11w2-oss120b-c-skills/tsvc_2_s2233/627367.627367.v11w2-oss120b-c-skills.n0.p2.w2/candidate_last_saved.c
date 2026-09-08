#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Compute column-wise prefix sum for aa (dependence across j)
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t i = 8; i < LEN_2D; ++i) {
            for (int64_t j = 8; j < LEN_2D; ++j) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
            }
        }
        // Compute row-wise prefix sum for bb (dependence across i)
        #pragma omp for schedule(static)
        for (int64_t j = 8; j < LEN_2D; ++j) {
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
            }
        }
    }
}
