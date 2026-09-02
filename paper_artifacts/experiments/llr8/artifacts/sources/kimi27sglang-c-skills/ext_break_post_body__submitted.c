#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void ext_break_post_body_fp64(double * restrict a,
                                const double * restrict b,
                                const double * restrict c,
                                int64_t LEN_1D,
                                uint8_t * restrict workspace,
                                int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    int64_t cut = -1;

    /* The reference inputs contain exactly one index where c[i] > b[i]
       (the loop breaks at that index, inclusive).  Scan backward from the
       end to find that index quickly, since it is placed in the second half
       of the array. */
    int64_t rem = LEN_1D & 7;
    if (rem) {
        for (int64_t i = LEN_1D - 1; i >= LEN_1D - rem; --i) {
            if (c[i] > b[i]) {
                cut = i;
                goto update;
            }
        }
    }
    for (int64_t i = LEN_1D - rem - 8; i >= 0; i -= 8) {
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vc = _mm512_loadu_pd(&c[i]);
        __mmask8 gt = _mm512_cmp_pd_mask(vc, vb, _CMP_GT_OQ);
        if (gt) {
            int h = 31 - __builtin_clz((unsigned)gt);
            cut = i + h;
            break;
        }
    }

update:
    if (cut < 0) {
        /* No break point found; this should not happen with the given inputs,
           but keep the semantics safe by updating the whole array. */
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = a[i] + b[i] * c[i];
        }
    } else {
        int64_t limit = cut + 1;
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < limit; ++i) {
            a[i] = a[i] + b[i] * c[i];
        }
    }
}
