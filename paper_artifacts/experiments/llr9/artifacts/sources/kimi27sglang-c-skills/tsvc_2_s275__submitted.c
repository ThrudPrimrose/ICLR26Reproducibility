#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double * restrict aa, double * restrict bb, double * restrict cc, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t n = LEN_2D;
    if (n <= 1) return;

    uint8_t *active = __builtin_alloca(n);
    for (int64_t i = 0; i < n; ++i) {
        active[i] = (aa[i] > 0.0) ? 1 : 0;
    }

    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t base_chunk = (n + nt - 1) / nt;
        const int64_t aligned_chunk = (base_chunk + 7) & ~((int64_t)7);
        int64_t i0 = tid * aligned_chunk;
        int64_t i1 = i0 + aligned_chunk;
        if (i1 > n) i1 = n;
        if (i0 > n) i0 = n;

        for (int64_t j = 1; j < n; ++j) {
            const int64_t row = j * n;
            const int64_t prev = (j - 1) * n;
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                if (active[i]) {
                    aa[row + i] = aa[prev + i] + bb[row + i] * cc[row + i];
                }
            }
        }
    }
}
