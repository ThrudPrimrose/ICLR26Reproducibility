/* Optimized TSVC tsvc_2 s311: parallel (OpenMP) + AVX-512 vectorized sum. */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#if defined(__AVX512F__)

static double chunk_sum(const double *a, int64_t n, int align64) {
  int64_t main32 = (n >> 5) << 5;
  __m512d s0, s1, s2, s3;
  if (align64) {
    s0 = _mm512_setzero_pd();
    s1 = _mm512_setzero_pd();
    s2 = _mm512_setzero_pd();
    s3 = _mm512_setzero_pd();
    for (int64_t i = 0; i < main32; i += 32) {
      s0 = _mm512_add_pd(s0, _mm512_load_pd(a + i));
      s1 = _mm512_add_pd(s1, _mm512_load_pd(a + i + 8));
      s2 = _mm512_add_pd(s2, _mm512_load_pd(a + i + 16));
      s3 = _mm512_add_pd(s3, _mm512_load_pd(a + i + 24));
    }
    s0 = _mm512_add_pd(s0, _mm512_add_pd(s1, s2));
    s0 = _mm512_add_pd(s0, s3);
    double r = _mm512_reduce_add_pd(s0);
    for (int64_t i = main32; i < n; i++) r += a[i];
    return r;
  } else {
    s0 = _mm512_setzero_pd();
    s1 = _mm512_setzero_pd();
    s2 = _mm512_setzero_pd();
    s3 = _mm512_setzero_pd();
    for (int64_t i = 0; i < main32; i += 32) {
      s0 = _mm512_add_pd(s0, _mm512_loadu_pd(a + i));
      s1 = _mm512_add_pd(s1, _mm512_loadu_pd(a + i + 8));
      s2 = _mm512_add_pd(s2, _mm512_loadu_pd(a + i + 16));
      s3 = _mm512_add_pd(s3, _mm512_loadu_pd(a + i + 24));
    }
    s0 = _mm512_add_pd(s0, _mm512_add_pd(s1, s2));
    s0 = _mm512_add_pd(s0, s3);
    double r = _mm512_reduce_add_pd(s0);
    for (int64_t i = main32; i < n; i++) r += a[i];
    return r;
  }
}

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
  if (LEN_1D <= 0) { sum_out[0] = 0.0; return; }
  int64_t n = LEN_1D;
  int64_t main32 = (n >> 5) << 5;
  int align64 = ((uintptr_t)a % 64) == 0;
  int64_t total_bytes = (int64_t)n * 8;
  int nt = omp_get_max_threads();
  /* use threads only when there is enough data; aim >= ~256KB per thread */
  while (nt > 1 && total_bytes / nt < (256LL << 20)) nt--;
  double total = 0.0;
  if (nt > 1 && main32 >= 32 * nt) {
    int64_t per = main32 / nt;
    if (per < 32) { per = 32; }
    #pragma omp parallel num_threads(nt) reduction(+:total)
    {
      int64_t rank = omp_get_thread_num();
      int64_t start = rank * per;
      int64_t len = per;
      if (rank == nt - 1) len = main32 - start;
      total += chunk_sum(a + start, len, align64);
    }
  } else {
    total = chunk_sum(a, main32, align64);
  }
  for (int64_t i = main32; i < n; i++) total += a[i];
  sum_out[0] = total;
}

#else

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
  double s = 0.0;
  for (int64_t i = 0; i < LEN_1D; i++) s += a[i];
  sum_out[0] = s;
}

#endif
