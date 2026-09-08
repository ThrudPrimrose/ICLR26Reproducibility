#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  const int nt = omp_get_max_threads();
  const int64_t chunk = (LEN_1D + nt - 1) / nt;

#pragma omp parallel for schedule(static)
  for (int t = 0; t < nt; ++t) {
    int64_t i = t * chunk;
    const int64_t end = (i + chunk < LEN_1D) ? i + chunk : LEN_1D;

    while (i < end && (((uintptr_t)(a + i)) & 63) != 0) {
      a[i] = b[ip[i]];
      ++i;
    }

    const int64_t vec_end = end - 7;
    for (; i < vec_end; i += 8) {
      for (int k = 0; k < 8; ++k) {
        _mm_prefetch((const char*)&b[ip[i + 56 + k]], _MM_HINT_T0);
      }
      __m256i idx = _mm256_loadu_si256((const __m256i*)(ip + i));
      __m512d val = _mm512_i32gather_pd(idx, (const void*)b, 8);
      _mm512_stream_pd(a + i, val);
    }

    for (; i < end; ++i) {
      a[i] = b[ip[i]];
    }
  }
}
