#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

#define CHUNK8(idx, dst) do { \
    __m512d a0 = _mm512_loadu_pd(&a[(idx) - 1]); \
    __m512d a1 = _mm512_loadu_pd(&a[(idx)]); \
    __m512d a2 = _mm512_loadu_pd(&a[(idx) + 1]); \
    __m512d a3 = _mm512_loadu_pd(&a[(idx) + 2]); \
    __m512d r  = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(a0, a1), a2), \
                                _mm512_add_pd(_mm512_add_pd(a1, a2), a3)); \
    _mm512_stream_pd((dst), r); \
} while (0)

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         int64_t LEN_1D,
                                         uint8_t *restrict workspace,
                                         int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const int64_t first = 1;
    const int64_t last = LEN_1D - 2;

    if (last <= first) {
        return;
    }

    if ((((uintptr_t)out) & 63) == 0) {
        const int64_t head_end = 8 < last ? 8 : last;
        #pragma omp simd
        for (int64_t i = first; i < head_end; i++) {
            double t0 = (a[i - 1] + a[i]) + a[i + 1];
            double t1 = (a[i] + a[i + 1]) + a[i + 2];
            out[i] = t0 * t1;
        }

        const int64_t limit = last - 31;
        if (head_end < limit) {
            #pragma omp parallel proc_bind(close)
            {
                #pragma omp for schedule(static) nowait
                for (int64_t i = head_end; i < limit; i += 32) {
                    CHUNK8(i,      &out[i]);
                    CHUNK8(i + 8,  &out[i + 8]);
                    CHUNK8(i + 16, &out[i + 16]);
                    CHUNK8(i + 24, &out[i + 24]);
                    if (i + 96 < last) {
                        _mm_prefetch((const char *)&a[i + 96], _MM_HINT_T0);
                    }
                }
                _mm_sfence();
            }
        }

        const int64_t tail_start = head_end < limit ? limit : head_end;
        #pragma omp simd
        for (int64_t i = tail_start; i < last; i++) {
            double t0 = (a[i - 1] + a[i]) + a[i + 1];
            double t1 = (a[i] + a[i + 1]) + a[i + 2];
            out[i] = t0 * t1;
        }
        return;
    }

    #pragma omp parallel for simd schedule(static)
    for (int64_t i = first; i < last; i++) {
        double t0 = (a[i - 1] + a[i]) + a[i + 1];
        double t1 = (a[i] + a[i + 1]) + a[i + 2];
        out[i] = t0 * t1;
    }
}
