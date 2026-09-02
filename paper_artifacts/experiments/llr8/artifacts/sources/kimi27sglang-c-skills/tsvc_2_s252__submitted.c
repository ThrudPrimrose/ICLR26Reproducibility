#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, double *restrict b, double *restrict c,
                      int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    if (LEN_1D <= 0) {
        return;
    }

    a[0] = b[0] * c[0];

    int64_t i = 1;

    // Advance to an output index whose address is 64-byte aligned.
    while (i < LEN_1D && (((uintptr_t)&a[i]) & 63)) {
        a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
        i++;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t k = i; k <= LEN_1D - 8; k += 8) {
        __m512d bi = _mm512_loadu_pd(b + k);
        __m512d ci = _mm512_loadu_pd(c + k);
        __m512d bim1 = _mm512_loadu_pd(b + k - 1);
        __m512d cim1 = _mm512_loadu_pd(c + k - 1);
        __m512d cur = _mm512_mul_pd(bi, ci);
        __m512d prev = _mm512_mul_pd(bim1, cim1);
        __m512d out = _mm512_add_pd(cur, prev);
        _mm512_stream_pd(a + k, out);
    }

    int64_t k_end = i + ((LEN_1D - i) & ~7);
    _mm_sfence();

    for (int64_t k = k_end; k < LEN_1D; k++) {
        a[k] = b[k] * c[k] + b[k - 1] * c[k - 1];
    }
}
