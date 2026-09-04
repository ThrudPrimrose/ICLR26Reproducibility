#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    int64_t n = LEN_1D;

    // For very small inputs just do the simple serial loop.
    if (n < 512) {
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
        return;
    }

    // Align output pointer to a 64-byte boundary for streaming stores.
    uintptr_t pa = (uintptr_t)a;
    int64_t align = ((64 - (pa & 63)) & 63) / sizeof(double);
    int64_t i = 0;
    for (; i < align; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }

    int64_t nvec = (n - i) & ~7;
    int64_t imax = i + nvec;

    #pragma omp parallel for schedule(static)
    for (int64_t j = i; j < imax; j += 8) {
        __m512d va = _mm512_loadu_pd(&a[j]);
        __m512d vb = _mm512_loadu_pd(&b[j]);
        __m512d vc = _mm512_loadu_pd(&c[j]);
        va = _mm512_mul_pd(va, vb);
        va = _mm512_mul_pd(va, vc);
        _mm512_stream_pd(&a[j], va);
    }

    _mm_sfence();

    for (int64_t j = imax; j < n; ++j) {
        a[j] = a[j] * b[j] * c[j];
    }
}
