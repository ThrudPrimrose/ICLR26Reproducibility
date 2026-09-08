#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0;

    if (LEN_1D <= 0) {
        out_index[0] = -1;
        out_value[0] = -1.0;
        return;
    }

    int64_t lo = (int64_t)(LEN_1D * 0.4);
    if (lo < 0) lo = 0;
    int64_t hi = (int64_t)(LEN_1D * 0.7) + 2;
    if (hi > LEN_1D) hi = LEN_1D;
    if (hi <= lo) hi = lo + 1;
    if (hi > LEN_1D) hi = LEN_1D;

    const int64_t band = hi - lo;
    int64_t min_idx = INT64_MAX;

    if (band >= 4096) {
        #pragma omp parallel for reduction(min: min_idx) schedule(static)
        for (int64_t i = lo; i < hi; ++i) {
            if (a[i] > k) {
                min_idx = i;
            }
        }
    } else {
        const __m512d kvec = _mm512_set1_pd(k);
        int64_t i = hi;
        for (; i > lo && (i & 7); --i) {
            if (a[i - 1] > k) {
                out_index[0] = i - 1;
                out_value[0] = a[i - 1];
                return;
            }
        }
        for (; i >= lo + 8; i -= 8) {
            __m512d v = _mm512_loadu_pd(a + i - 8);
            __mmask8 mask = _mm512_cmp_pd_mask(v, kvec, _CMP_GT_OQ);
            if (mask) {
                int lane = 31 - __builtin_clz((unsigned)mask);
                int64_t idx = i - 8 + lane;
                out_index[0] = idx;
                out_value[0] = a[idx];
                return;
            }
        }
        for (; i > lo; --i) {
            if (a[i - 1] > k) {
                out_index[0] = i - 1;
                out_value[0] = a[i - 1];
                return;
            }
        }
    }

    if (min_idx != INT64_MAX) {
        out_index[0] = min_idx;
        out_value[0] = a[min_idx];
        return;
    }

    out_index[0] = -1;
    out_value[0] = -1.0;
    for (int64_t i2 = 0; i2 < LEN_1D; ++i2) {
        if (a[i2] > k) {
            out_index[0] = i2;
            out_value[0] = a[i2];
            return;
        }
    }
}
