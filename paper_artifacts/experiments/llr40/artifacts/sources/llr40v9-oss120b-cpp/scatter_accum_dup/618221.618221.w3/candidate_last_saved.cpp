#include <cstdint>
#include <omp.h>

extern "C" void scatter_accum_dup_fp64(const double* __restrict__ src,
                                      const int32_t* __restrict__ ip,
                                      double* __restrict__ bins,
                                      const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    #pragma omp parallel for schedule(static) ordered
    for (int64_t i = 0; i < LEN_1D; ++i) {
        #pragma omp ordered
        bins[ip[i]] += src[i];
    }
}
