#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void compact_threshold_pack(const double *restrict src,
                            const double *restrict weight,
                            double *restrict packed,
                            int64_t *restrict out_count,
                            int64_t LEN_1D)
{
    int64_t n = 0;
    #pragma omp parallel for reduction(inscan,+:n)
    for (int64_t i = 0; i < LEN_1D; i++) {
        #pragma omp scan exclusive(n)
        if (src[i] > 0.0) {
            packed[n] = src[i] * weight[i];
            n += 1;
        }
    }
    out_count[0] = n;
}
