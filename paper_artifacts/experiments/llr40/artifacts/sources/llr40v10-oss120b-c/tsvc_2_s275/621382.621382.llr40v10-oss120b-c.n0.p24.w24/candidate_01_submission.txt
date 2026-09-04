#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double prev = aa[i]; // aa[0, i]
        if (prev > 0.0) {
            int64_t offset = i;
            for (int64_t j = 1; j < LEN_2D; ++j) {
                offset += LEN_2D;
                double prod = bb[offset] * cc[offset];
                double cur = prev + prod;
                aa[offset] = cur;
                prev = cur;
            }
        }
    }
}
