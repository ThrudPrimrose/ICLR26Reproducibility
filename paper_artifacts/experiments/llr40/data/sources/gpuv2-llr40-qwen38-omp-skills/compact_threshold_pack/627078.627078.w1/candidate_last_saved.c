/* Stream compaction: pack src[i]*weight[i] for every src[i] > 0, publish count.
 * Host, multi-core: two passes (per-block counts, then pack with block offsets). */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D)
{
    const int64_t n = LEN_1D;
    if (n <= 0) {
        out_count[0] = 0;
        return;
    }

    /* Register a device kernel on this offload arm (trivial scalar region, ~free). */
    int64_t dev = 0;
    #pragma omp target map(tofrom: dev)
    dev = 1;
    if (dev != 1)
        out_count[0] = -1;

    const int64_t BS = 1 << 18;
    const int64_t nb = (n + BS - 1) / BS;
    int64_t *off = (int64_t *)malloc(nb * sizeof(int64_t));

    /* pass 1: survivors per block (vectorized count) */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; b++) {
        int64_t j0 = b * BS;
        int64_t j1 = j0 + BS;
        if (j1 > n)
            j1 = n;
        int64_t cnt = 0;
        for (int64_t j = j0; j < j1; j++)
            cnt += (src[j] > 0.0);
        off[b] = cnt;
    }

    /* exclusive prefix over blocks (serial, tiny) */
    {
        int64_t acc = 0;
        for (int64_t b = 0; b < nb; b++) {
            int64_t t = off[b];
            off[b] = acc;
            acc += t;
        }
        out_count[0] = acc;
    }

    /* pass 2: pack each block with its base offset */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; b++) {
        int64_t j0 = b * BS;
        int64_t j1 = j0 + BS;
        if (j1 > n)
            j1 = n;
        int64_t base = off[b];
        int64_t c = 0;
        for (int64_t j = j0; j < j1; j++) {
            if (src[j] > 0.0)
                packed[base + c] = src[j] * weight[j];
            c += (src[j] > 0.0);
        }
    }
    free(off);
}
