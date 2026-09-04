#include <stdint.h>
#include <omp.h>

#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
void scatter_accum_dup_fp64(double *bins, const uint32_t *ip, const double *src, const int64_t LEN_1D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        uint32_t idx = ip[i];
        #pragma omp atomic
        bins[idx] += src[i];
    }
}
#pragma GCC pop_options
