#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s255_fp64(double* restrict a, double* restrict b, int64_t LEN_1D,
                      uint8_t* restrict workspace, int64_t workspace_bytes) {
    const double scale = 0.333;
    const __m512d vscale = _mm512_set1_pd(scale);
    const __m512i idx0 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);
    const __m512i idx1 = _mm512_setr_epi64(7,8,9,10,11,12,13,14);

    if (LEN_1D <= 0) return;

    double bn1 = b[LEN_1D - 1];
    double bn2 = b[LEN_1D - 2];

    if (LEN_1D >= 1) {
        a[0] = (b[0] + bn1 + bn2) * scale;
    }
    if (LEN_1D >= 2) {
        a[1] = (b[1] + b[0] + bn1) * scale;
    }
    if (LEN_1D <= 2) return;

    #pragma omp parallel default(none) shared(a, b, LEN_1D, scale, vscale, idx0, idx1)
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t s = (int64_t)tid * LEN_1D / nt;
        int64_t e = (int64_t)(tid + 1) * LEN_1D / nt;
        if (s < 2) s = 2;

        int64_t i = s;
        for (; i < e && (i & 7); ++i) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * scale;
        }

        if (i + 8 <= e) {
            __m512d low = _mm512_load_pd(b + i - 8);
            for (; i + 8 <= e; i += 8) {
                _mm_prefetch((const char*)(b + i + 64), _MM_HINT_NTA);
                __m512d high = _mm512_load_pd(b + i);
                __m512d v0 = _mm512_permutex2var_pd(low, idx0, high);
                __m512d v1 = _mm512_permutex2var_pd(low, idx1, high);
                __m512d sum = _mm512_add_pd(v0, _mm512_add_pd(v1, high));
                sum = _mm512_mul_pd(sum, vscale);
                _mm512_stream_pd(a + i, sum);
                low = high;
            }
        }

        for (; i < e; ++i) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * scale;
        }
    }
    _mm_sfence();
}
