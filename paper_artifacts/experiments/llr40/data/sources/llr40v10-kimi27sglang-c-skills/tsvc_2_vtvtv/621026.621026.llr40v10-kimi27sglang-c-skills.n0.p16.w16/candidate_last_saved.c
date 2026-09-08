#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#define DIST1 48
#define DIST2 96

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t i = tid * chunk;
        const int64_t end = (i + chunk < LEN_1D) ? (i + chunk) : LEN_1D;

        while (i < end && ((uintptr_t)(a + i) & 63)) {
            a[i] = a[i] * b[i] * c[i];
            ++i;
        }

        for (; i + 7 < end; i += 8) {
            _mm_prefetch((const char*)(a + i + DIST1), _MM_HINT_NTA);
            _mm_prefetch((const char*)(b + i + DIST1), _MM_HINT_NTA);
            _mm_prefetch((const char*)(c + i + DIST1), _MM_HINT_NTA);
            _mm_prefetch((const char*)(a + i + DIST2), _MM_HINT_NTA);
            _mm_prefetch((const char*)(b + i + DIST2), _MM_HINT_NTA);
            _mm_prefetch((const char*)(c + i + DIST2), _MM_HINT_NTA);
            __m512d va = _mm512_load_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vc = _mm512_loadu_pd(c + i);
            va = _mm512_mul_pd(va, vb);
            va = _mm512_mul_pd(va, vc);
            _mm512_store_pd(a + i, va);
        }

        for (; i < end; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
    }
}
