#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  const int64_t PF_DIST = 128;
  int64_t i = 0;

  while (i < LEN_1D && (((uintptr_t)&a[i]) & 15)) {
    a[i] = b[ip[i]];
    ++i;
  }

  const int64_t lim = LEN_1D - 1;
  #pragma omp parallel for schedule(static)
  for (int64_t j = i; j < lim; j += 2) {
    if (j + PF_DIST < LEN_1D) {
      _mm_prefetch((const char *)&b[ip[j + PF_DIST]], _MM_HINT_T0);
    }
    __m128d v = _mm_set_pd(b[ip[j + 1]], b[ip[j]]);
    _mm_stream_pd(&a[j], v);
  }

  if ((LEN_1D - i) & 1) {
    a[LEN_1D - 1] = b[ip[LEN_1D - 1]];
  }
}
