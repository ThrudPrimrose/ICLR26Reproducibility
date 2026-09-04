#include <stdint.h>
#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
static inline void nt_row(double *dst, const double *sr, int64_t N, int mul2) {
  int64_t j = 0;
  uintptr_t off = (uintptr_t)dst & 63;
  int64_t h = off ? ((64 - off) >> 3) : 0;
  if (h > N) h = N;
  if (mul2) { for (; j < h; ++j) dst[j] = sr[j] * 2.0; }
  else      { for (; j < h; ++j) dst[j] = sr[j] + 1.0; }
  int64_t mid = (N - j) & ~7;
  if (mul2) {
    const __m512d two = _mm512_set1_pd(2.0);
    for (; j < mid; j += 8)
      _mm512_stream_pd(dst + j, _mm512_mul_pd(_mm512_loadu_pd(sr + j), two));
    for (; j < N; ++j) dst[j] = sr[j] * 2.0;
  } else {
    const __m512d one = _mm512_set1_pd(1.0);
    for (; j < mid; j += 8)
      _mm512_stream_pd(dst + j, _mm512_add_pd(_mm512_loadu_pd(sr + j), one));
    for (; j < N; ++j) dst[j] = sr[j] + 1.0;
  }
}
#endif

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int wb = (K > 0);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) {
    const int64_t base = i * N;
    const double *sr = src + base;
    const int wa = (cond[i] > 0.0);
#if defined(__AVX512F__)
    if (wb) nt_row(b + base, sr, N, 0);
    if (wa) nt_row(a + base, sr, N, 1);
#else
    if (wb) {
      double *bp = b + base;
      for (int64_t j = 0; j < N; ++j) bp[j] = sr[j] + 1.0;
    }
    if (wa) {
      double *ap = a + base;
      for (int64_t j = 0; j < N; ++j) ap[j] = sr[j] * 2.0;
    }
#endif
  }
}
