#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define NT 48

__attribute__((constructor)) static void tsvc_2_s311_warm_pool(void) {
  #pragma omp parallel num_threads(NT)
  {
  }
}

static double chunk_sum(const double *restrict r, int64_t m) {
  int64_t i = 0;
  __m512d v0 = _mm512_setzero_pd(), v1 = _mm512_setzero_pd();
  __m512d v2 = _mm512_setzero_pd(), v3 = _mm512_setzero_pd();
  int64_t limit = m & ~31LL;
  for (; i < limit; i += 32) {
    v0 = _mm512_add_pd(v0, _mm512_loadu_pd(r + i));
    v1 = _mm512_add_pd(v1, _mm512_loadu_pd(r + i + 8));
    v2 = _mm512_add_pd(v2, _mm512_loadu_pd(r + i + 16));
    v3 = _mm512_add_pd(v3, _mm512_loadu_pd(r + i + 24));
  }
  for (; i + 8 <= m; i += 8) {
    v0 = _mm512_add_pd(v0, _mm512_loadu_pd(r + i));
  }
  __m512d v = v0 + v1 + v2 + v3;
  __m256d h = _mm512_castpd512_pd256(v);
  h = _mm256_add_pd(h, _mm512_extractf64x4_pd(v, 1));
  __m128d lo = _mm256_castpd256_pd128(h);
  lo = _mm_add_pd(lo, _mm256_extractf128_pd(h, 1));
  lo = _mm_hadd_pd(lo, lo);
  double s = _mm_cvtsd_f64(lo);
  for (; i < m; i++) s += r[i];
  return s;
}

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
  int64_t n = LEN_1D;
  int64_t pro = (int)(((32 - ((uintptr_t)a & 31)) >> 3) & 3);
  if (n < (1 << 22)) {
    double s = 0.0;
    int64_t i = 0;
    for (; i < pro && i < n; i++) s += a[i];
    s += chunk_sum(a + pro, n - pro);
    sum_out[0] = s;
    return;
  }
  double total = 0.0;
  int64_t cap = omp_get_max_threads();
  if ((int64_t)NT < cap) cap = NT;
  #pragma omp parallel num_threads((int)cap) reduction(+:total)
  {
    int64_t nt = omp_get_num_threads(), tid = omp_get_thread_num();
    int64_t s = (n * tid) / nt;
    int64_t e = (n * (tid + 1)) / nt;
    int64_t i = s;
    double local = 0.0;
    int64_t a4 = s + ((pro - s) & 3);
    for (; i < a4 && i < e; i++) local += a[i];
    if (a4 < e) local += chunk_sum(a + a4, e - a4);
    total += local;
  }
  sum_out[0] = total;
}
