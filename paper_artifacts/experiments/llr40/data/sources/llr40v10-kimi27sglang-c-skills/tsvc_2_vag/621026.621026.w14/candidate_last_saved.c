#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const int64_t main_end = LEN_1D & ~7LL;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < main_end; i += 8) {
        __m256i idx_future = _mm256_loadu_si256((const __m256i*)(ip + i + 256));
        _mm_prefetch((const char*)(ip + i + 512), _MM_HINT_T0);
        _mm512_prefetch_i32gather_pd(idx_future, b, 8, _MM_HINT_T0);
        __m256i idx = _mm256_loadu_si256((const __m256i*)(ip + i));
        __m512d val = _mm512_i32gather_pd(idx, b, 8);
        _mm512_storeu_pd(a + i, val);
    }
    for (int64_t i = main_end; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
