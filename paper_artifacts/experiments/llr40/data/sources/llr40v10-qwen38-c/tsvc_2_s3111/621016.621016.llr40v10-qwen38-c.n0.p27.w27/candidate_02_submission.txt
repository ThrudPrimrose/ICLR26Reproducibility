#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* Sum of positive elements.  (a[i] > 0.0) ? a[i] : 0.0 == max(a[i], 0.0),
 * including NaN (VMAXPD returns the non-NaN operand).  Each thread accumulates
 * into 8 independent vector lanes so the inner loop has no serial FP chain. */

static inline double hsum512(__m512d v) { return _mm512_reduce_add_pd(v); }

__attribute__((target("avx512f")))
static void reduce_avx512(const double *restrict a, const int64_t n, double *out) {
  const __m512d z = _mm512_setzero_pd();
  __m512d s0 = z, s1 = z, s2 = z, s3 = z, s4 = z, s5 = z, s6 = z, s7 = z;
  int64_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m512d v0 = _mm512_loadu_pd(a + i + 0);
    __m512d v1 = _mm512_loadu_pd(a + i + 8);
    __m512d v2 = _mm512_loadu_pd(a + i + 16);
    __m512d v3 = _mm512_loadu_pd(a + i + 24);
    __m512d v4 = _mm512_loadu_pd(a + i + 32);
    __m512d v5 = _mm512_loadu_pd(a + i + 40);
    __m512d v6 = _mm512_loadu_pd(a + i + 48);
    __m512d v7 = _mm512_loadu_pd(a + i + 56);
    s0 = _mm512_add_pd(s0, _mm512_max_pd(v0, z));
    s1 = _mm512_add_pd(s1, _mm512_max_pd(v1, z));
    s2 = _mm512_add_pd(s2, _mm512_max_pd(v2, z));
    s3 = _mm512_add_pd(s3, _mm512_max_pd(v3, z));
    s4 = _mm512_add_pd(s4, _mm512_max_pd(v4, z));
    s5 = _mm512_add_pd(s5, _mm512_max_pd(v5, z));
    s6 = _mm512_add_pd(s6, _mm512_max_pd(v6, z));
    s7 = _mm512_add_pd(s7, _mm512_max_pd(v7, z));
  }
  __m512d s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  double r = hsum512(s);
  for (; i < n; ++i)
    if (a[i] > 0.0) r += a[i];
  *out = r;
}

__attribute__((target("avx2")))
static void reduce_avx2(const double *restrict a, const int64_t n, double *out) {
  const __m256d z = _mm256_setzero_pd();
  __m256d s0 = z, s1 = z, s2 = z, s3 = z;
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256d v0 = _mm256_loadu_pd(a + i + 0);
    __m256d v1 = _mm256_loadu_pd(a + i + 4);
    __m256d v2 = _mm256_loadu_pd(a + i + 8);
    __m256d v3 = _mm256_loadu_pd(a + i + 12);
    __m256d v4 = _mm256_loadu_pd(a + i + 16);
    __m256d v5 = _mm256_loadu_pd(a + i + 20);
    __m256d v6 = _mm256_loadu_pd(a + i + 24);
    __m256d v7 = _mm256_loadu_pd(a + i + 28);
    s0 = _mm256_add_pd(s0, _mm256_max_pd(v0, z));
    s1 = _mm256_add_pd(s1, _mm256_max_pd(v1, z));
    s2 = _mm256_add_pd(s2, _mm256_max_pd(v2, z));
    s3 = _mm256_add_pd(s3, _mm256_max_pd(v3, z));
    s0 = _mm256_add_pd(s0, _mm256_max_pd(v4, z));
    s1 = _mm256_add_pd(s1, _mm256_max_pd(v5, z));
    s2 = _mm256_add_pd(s2, _mm256_max_pd(v6, z));
    s3 = _mm256_add_pd(s3, _mm256_max_pd(v7, z));
  }
  __m256d s = (s0 + s1) + (s2 + s3);
  double t[4];
  _mm256_storeu_pd(t, s);
  double r = (t[0] + t[1]) + (t[2] + t[3]);
  for (; i < n; ++i)
    if (a[i] > 0.0) r += a[i];
  *out = r;
}

static void reduce_scalar(const double *restrict a, const int64_t n, double *out) {
  double r = 0.0;
  for (int64_t i = 0; i < n; ++i)
    if (a[i] > 0.0) r += a[i];
  *out = r;
}

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  double sum = 0.0;
#pragma omp parallel reduction(+:sum)
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t per = (LEN_1D + nt - 1) / nt;
    const int64_t lo = (int64_t)tid * per;
    int64_t hi = lo + per;
    if (hi > LEN_1D) hi = LEN_1D;
    double s = 0.0;
    if (hi > lo) {
      if (__builtin_cpu_supports("avx512f"))
        reduce_avx512(a + lo, hi - lo, &s);
      else if (__builtin_cpu_supports("avx2"))
        reduce_avx2(a + lo, hi - lo, &s);
      else
        reduce_scalar(a + lo, hi - lo, &s);
    }
    sum += s;
  }
  b[0] = sum;
}
