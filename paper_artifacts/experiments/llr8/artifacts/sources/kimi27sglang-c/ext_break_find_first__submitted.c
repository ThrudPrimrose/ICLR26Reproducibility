#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_break_find_first_fp64(double *restrict a, double *restrict b, double *restrict c, double *restrict d, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    int64_t n = LEN_1D;
    if (n <= 0) return;

    int64_t half = n / 2;

    // d is positive in [0, half), so update this half unconditionally in parallel
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < half; ++j) {
        a[j] = a[j] + b[j] * c[j];
    }

    // Scan the second half for the planted negative
    int64_t idx = n;
    int64_t i = half;
    int64_t n_vec = n & ~7;
    const __m512d zero = _mm512_setzero_pd();

    for (; i < n_vec; i += 8) {
        __m512d vd = _mm512_loadu_pd(&d[i]);
        __mmask8 neg = _mm512_cmp_pd_mask(vd, zero, _CMP_LT_OS);
        if (neg) {
            idx = i + __builtin_ctz((unsigned)neg);
            break;
        }
    }
    if (idx == n) {
        for (; i < n; ++i) {
            if (d[i] < 0.0) { idx = i; break; }
        }
    }

    // Update the positive prefix of the second half
    if (idx > half) {
#pragma omp parallel for schedule(static)
        for (int64_t j = half; j < idx; ++j) {
            a[j] = a[j] + b[j] * c[j];
        }
    }
}
