#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include <stdbool.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    const __m256d vs = _mm256_set1_pd(s);
    const int64_t vec_end = LEN_1D - (LEN_1D % 4);
    uintptr_t a_addr = (uintptr_t) a;
    uintptr_t b_addr = (uintptr_t) b;
    bool aligned = ((a_addr % 32) == 0) && ((b_addr % 32) == 0);
    if (aligned) {
        #pragma omp parallel for if(LEN_1D > 4096) schedule(static)
        for (int64_t i = 0; i < vec_end; i += 4) {
            if (i + 16 < vec_end) {
                _mm_prefetch((const char*)(a + i + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + i + 16), _MM_HINT_T0);
            }
            __m256d va = _mm256_load_pd(a + i);
            __m256d vb = _mm256_load_pd(b + i);
            __m256d res = _mm256_fmadd_pd(vb, vs, va);
            _mm256_store_pd(a + i, res);
        }
    } else {
        #pragma omp parallel for if(LEN_1D > 4096) schedule(static)
        for (int64_t i = 0; i < vec_end; i += 4) {
            if (i + 16 < vec_end) {
                _mm_prefetch((const char*)(a + i + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + i + 16), _MM_HINT_T0);
            }
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d res = _mm256_fmadd_pd(vb, vs, va);
            _mm256_storeu_pd(a + i, res);
        }
    }
    for (int64_t i = vec_end; i < LEN_1D; ++i) {
        a[i] += b[i] * s;
    }
}
