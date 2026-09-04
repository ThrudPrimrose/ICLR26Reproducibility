#include <stdint.h>
#include <immintrin.h>
#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {

    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t jmax = i / VLEN;
        if (jmax >= LEN_2D) jmax = LEN_2D - 1;
        const int64_t base = i * LEN_2D;
        int64_t j = 0;

        const int can_stream = (jmax >= 7) && (((uintptr_t)&aa[base]) % 64 == 0);

        if (can_stream) {
            for (; j + 7 <= jmax; j += 8) {
                __m512d b = _mm512_loadu_pd(&bb[base + j]);
                __m512d c = _mm512_loadu_pd(&cc[base + j]);
                __m512d a = _mm512_add_pd(b, c);
                _mm512_stream_pd(&aa[base + j], a);
            }
        } else {
            for (; j + 7 <= jmax; j += 8) {
                __m512d b = _mm512_loadu_pd(&bb[base + j]);
                __m512d c = _mm512_loadu_pd(&cc[base + j]);
                __m512d a = _mm512_add_pd(b, c);
                _mm512_storeu_pd(&aa[base + j], a);
            }
        }

        for (; j <= jmax; ++j) {
            aa[base + j] = bb[base + j] + cc[base + j];
        }
    }
}
