#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; i += 8) {
        if (i + 7 < LEN_1D) {
            __m256i idx = _mm256_loadu_si256((const __m256i *)&ip[i]);
            __m512d vals = _mm512_i32gather_pd(idx, b, 8);
            _mm512_stream_pd(&a[i], vals);
        } else {
            for (int64_t j = i; j < LEN_1D; ++j) {
                a[j] = b[ip[j]];
            }
        }
    }
    _mm_sfence();
}
