#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
#define VNT_ROW(mul2)                                                        \
  int64_t j = 0;                                                             \
  const uintptr_t off_ = (uintptr_t)dst & 63;                                \
  int64_t h_ = off_ ? ((64 - off_) >> 3) : 0;                                \
  if (h_ > N) h_ = N;                                                        \
  if (mul2) { for (; j < h_; ++j) dst[j] = sr[j] * 2.0; }                    \
  else      { for (; j < h_; ++j) dst[j] = sr[j] + 1.0; }                    \
  int64_t mid_ = (N - j) & ~7;                                               \
  if (mul2) {                                                                \
    for (; j + 16 <= mid_; j += 16) {                                        \
      const __m512d v0 = _mm512_loadu_pd(sr + j);                            \
      const __m512d v1 = _mm512_loadu_pd(sr + j + 8);                        \
      _mm512_stream_pd(dst + j, _mm512_mul_pd(v0, _mm512_set1_pd(2.0)));     \
      _mm512_stream_pd(dst + j + 8, _mm512_mul_pd(v1, _mm512_set1_pd(2.0))); \
    }                                                                        \
    for (; j < mid_; j += 8) {                                               \
      _mm512_stream_pd(dst + j, _mm512_mul_pd(_mm512_loadu_pd(sr + j), _mm512_set1_pd(2.0))); \
    }                                                                        \
    for (; j < N; ++j) dst[j] = sr[j] * 2.0;                                \
  } else {                                                                   \
    for (; j + 16 <= mid_; j += 16) {                                        \
      const __m512d v0 = _mm512_loadu_pd(sr + j);                            \
      const __m512d v1 = _mm512_loadu_pd(sr + j + 8);                        \
      _mm512_stream_pd(dst + j, _mm512_add_pd(v0, _mm512_set1_pd(1.0)));     \
      _mm512_stream_pd(dst + j + 8, _mm512_add_pd(v1, _mm512_set1_pd(1.0))); \
    }                                                                        \
    for (; j < mid_; j += 8) {                                               \
      _mm512_stream_pd(dst + j, _mm512_add_pd(_mm512_loadu_pd(sr + j), _mm512_set1_pd(1.0))); \
    }                                                                        \
    for (; j < N; ++j) dst[j] = sr[j] + 1.0;                                \
  }

#define VNT_ROW2()                                                             \
  int64_t j = 0;                                                               \
  const uintptr_t off_ = (uintptr_t)da_ & 63;                                  \
  int64_t h_ = off_ ? ((64 - off_) >> 3) : 0;                                  \
  if (h_ > N_) h_ = N_;                                                        \
  for (; j < h_; ++j) {                                                        \
    const double v = sr_[j];                                                   \
    da_[j] = v * 2.0;                                                          \
    db_[j] = v + 1.0;                                                          \
  }                                                                            \
  int64_t mid_ = (N_ - j) & ~7;                                                \
  for (; j + 16 <= mid_; j += 16) {                                            \
    const __m512d v0 = _mm512_loadu_pd(sr_ + j);                               \
    const __m512d v1 = _mm512_loadu_pd(sr_ + j + 8);                           \
    _mm512_stream_pd(da_ + j, _mm512_mul_pd(v0, _mm512_set1_pd(2.0)));         \
    _mm512_stream_pd(db_ + j, _mm512_add_pd(v0, _mm512_set1_pd(1.0)));         \
    _mm512_stream_pd(da_ + j + 8, _mm512_mul_pd(v1, _mm512_set1_pd(2.0)));     \
    _mm512_stream_pd(db_ + j + 8, _mm512_add_pd(v1, _mm512_set1_pd(1.0)));     \
  }                                                                            \
  for (; j < mid_; j += 8) {                                                   \
    const __m512d v = _mm512_loadu_pd(sr_ + j);                                \
    _mm512_stream_pd(da_ + j, _mm512_mul_pd(v, _mm512_set1_pd(2.0)));          \
    _mm512_stream_pd(db_ + j, _mm512_add_pd(v, _mm512_set1_pd(1.0)));          \
  }                                                                            \
  for (; j < N_; ++j) {                                                        \
    const double v = sr_[j];                                                   \
    da_[j] = v * 2.0;                                                          \
    db_[j] = v + 1.0;                                                          \
  }

static inline void nt_row(double *dst, const double *sr, int64_t N, int mul2) {
  VNT_ROW(mul2)
}

static inline void nt_row2(double *da_, double *db_, const double *sr_, int64_t N_) {
  VNT_ROW2()
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
    if (wb && wa) {
      if ((((uintptr_t)a ^ (uintptr_t)b) & 63) == 0)
        nt_row2(a + base, b + base, sr, N);
      else {
        nt_row(b + base, sr, N, 0);
        nt_row(a + base, sr, N, 1);
      }
    } else if (wb) {
      nt_row(b + base, sr, N, 0);
    } else if (wa) {
      nt_row(a + base, sr, N, 1);
    }
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
