#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define TWO _mm512_set1_pd(2.0)
#define ONE _mm512_set1_pd(1.0)

static inline void row64(const double *restrict srow, double *restrict arow, double *restrict brow,
                         int64_t n, int64_t off, int64_t je, int pos, int64_t K) {
  const double c2 = 2.0, c1 = 1.0;
  for (int64_t t = 0; t < off; ++t) {
    const double v = srow[t];
    if (pos) arow[t] = v * c2;
    if (K > 0) brow[t] = v + c1;
  }
  if (pos) {
    if (K > 0) {
      for (int64_t j = off; j < je; j += 8) {
        const __m512d v = _mm512_load_pd(srow + j);
        _mm512_stream_pd(arow + j, _mm512_mul_pd(v, TWO));
        _mm512_stream_pd(brow + j, _mm512_add_pd(v, ONE));
      }
    } else {
      for (int64_t j = off; j < je; j += 8) {
        const __m512d v = _mm512_load_pd(srow + j);
        _mm512_stream_pd(arow + j, _mm512_mul_pd(v, TWO));
      }
    }
  } else if (K > 0) {
    for (int64_t j = off; j < je; j += 8) {
      const __m512d v = _mm512_load_pd(srow + j);
      _mm512_stream_pd(brow + j, _mm512_add_pd(v, ONE));
    }
  }
  for (int64_t j = je; j < n; ++j) {
    const double v = srow[j];
    if (pos) arow[j] = v * c2;
    if (K > 0) brow[j] = v + c1;
  }
}

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 1024) {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t off = (int64_t)(-(int64_t)((size_t)(i * n) & 7)) & 7;
      const int64_t je = n - ((n - off) & 7);
      row64(src + i * n, a + i * n, b + i * n, n, off, je, (int)(cond[i] > 0.0), K);
    }
    return;
  }
#pragma omp parallel
  {
    const int64_t nt = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t c = (n + nt - 1) / nt;
    const int64_t i0 = tid * c, i1 = (i0 + c < n) ? i0 + c : n;
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t off = (int64_t)(-(int64_t)((size_t)(i * n) & 7)) & 7;
      const int64_t je = n - ((n - off) & 7);
      row64(src + i * n, a + i * n, b + i * n, n, off, je, (int)(cond[i] > 0.0), K);
    }
  }
}
