#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  int nt = omp_get_max_threads();
  #pragma omp parallel num_threads(nt)
  {
    int64_t chunk = (LEN_1D + nt - 1) / nt;
    int64_t s = (int64_t)omp_get_thread_num() * chunk;
    int64_t e = s + chunk; if (e > LEN_1D) e = LEN_1D;
    double *ar = a + s; const double *br = b; const int32_t *ipr = ip + s;
    int64_t n = e - s;
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
      __m512d v0,v1,v2,v3;
      { __m256i idx = _mm256_loadu_si256((const __m256i*)(ipr+i));
        __m512i off = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx),3);
        v0 = _mm512_i64gather_pd(off, br, 1); }
      { __m256i idx = _mm256_loadu_si256((const __m256i*)(ipr+i+8));
        __m512i off = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx),3);
        v1 = _mm512_i64gather_pd(off, br, 1); }
      { __m256i idx = _mm256_loadu_si256((const __m256i*)(ipr+i+16));
        __m512i off = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx),3);
        v2 = _mm512_i64gather_pd(off, br, 1); }
      { __m256i idx = _mm256_loadu_si256((const __m256i*)(ipr+i+24));
        __m512i off = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx),3);
        v3 = _mm512_i64gather_pd(off, br, 1); }
      _mm512_storeu_pd(ar+i, v0);
      _mm512_storeu_pd(ar+i+8, v1);
      _mm512_storeu_pd(ar+i+16, v2);
      _mm512_storeu_pd(ar+i+24, v3);
    }
    for (; i + 8 <= n; i += 8) {
      __m256i idx = _mm256_loadu_si256((const __m256i*)(ipr+i));
      __m512i off = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx),3);
      _mm512_storeu_pd(ar+i, _mm512_i64gather_pd(off, br, 1));
    }
    for (; i < n; ++i) ar[i] = br[(size_t)ipr[i]];
  }
}
