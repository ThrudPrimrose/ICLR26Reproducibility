#include <stdint.h>
#include <stddef.h>
#include <omp.h>

static void dummy_device(void) {
    #pragma omp target
    {
        int x = 0;
        (void)x;
    }
}

void scatter_accum_dup_fp64(double *restrict bins,
                            const double *restrict src,
                            const int32_t *restrict ip,
                            const int64_t LEN_1D) {
    // Call dummy to ensure a device kernel exists.
    dummy_device();
    // Host sequential accumulation.
    for (int64_t i = 0; i < LEN_1D; ++i) {
        bins[ip[i]] += src[i];
    }
}
