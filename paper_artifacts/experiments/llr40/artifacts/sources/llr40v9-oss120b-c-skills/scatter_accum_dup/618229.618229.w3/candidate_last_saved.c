#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src, const int64_t *restrict ip, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // Copy src to avoid aliasing issues.
    double *src_tmp = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!src_tmp) return;
    memcpy(src_tmp, src, (size_t)LEN_1D * sizeof(double));
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t rem = ip[i] % LEN_1D;
        if (rem < 0) rem += LEN_1D;
        size_t idx = (size_t)rem;
        bins[idx] += src_tmp[i];
    }
    free(src_tmp);
}
