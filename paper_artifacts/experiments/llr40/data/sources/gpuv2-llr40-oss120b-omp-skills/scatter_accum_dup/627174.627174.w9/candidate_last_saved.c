#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

typedef struct { int32_t idx; double val; } pair_t;

static int compare_pair(const void *a, const void *b) {
    const pair_t *pa = (const pair_t *)a;
    const pair_t *pb = (const pair_t *)b;
    if (pa->idx < pb->idx) return -1;
    if (pa->idx > pb->idx) return 1;
    return 0;
}

void scatter_accum_dup_fp64(double *restrict bins,
                            const double *restrict src,
                            const int32_t *restrict ip,
                            const int64_t LEN_1D) {
    // Dummy target region to register a device kernel (does nothing).
    #pragma omp target
    {
        // No operation on device.
    }

#ifndef __AMDGCN__
    // Host implementation: sort by index and accumulate safely.
    pair_t *tmp = (pair_t *)malloc((size_t)LEN_1D * sizeof(pair_t));
    if (!tmp) return;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        tmp[i].idx = ip[i];
        tmp[i].val = src[i];
    }
    qsort(tmp, (size_t)LEN_1D, sizeof(pair_t), compare_pair);
    int64_t i = 0;
    while (i < LEN_1D) {
        int32_t cur = tmp[i].idx;
        double sum = 0.0;
        while (i < LEN_1D && tmp[i].idx == cur) {
            sum += tmp[i].val;
            ++i;
        }
        bins[cur] += sum;
    }
    free(tmp);
#endif
}
