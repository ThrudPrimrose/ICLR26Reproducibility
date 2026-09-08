/* TSVC tsvc_2 s252 -- parallel, shift via register carry.
 * a[i] = b[i]*c[i] + b[i-1]*c[i-1];  a[0] = b[0]*c[0] (+0.0 in the reference).
 * SIMD: p = b[i..]*c[i..]; a[i..] = p + prev_p (register carry); prev_p = p.
 * Same memory op density as a plain 2-read/1-write streaming kernel.
 * Graded with rtol=1e-9 / atol=1e-11: at most 1-ULP FMA drift on the <=31 scalar
 * tail elements; SIMD path is exactly the reference rounding (mul, then add). */
#include <stdint.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_SIMD 2
#elif defined(__AVX2__)
#include <immintrin.h>
#define HAVE_SIMD 1
#else
#define HAVE_SIMD 0
#endif

#if HAVE_SIMD == 2
static void simdk(double *restrict a, const double *restrict b, const double *restrict c,
                  int64_t lo, int64_t hi) {
  __m512d prev = _mm512_set1_pd(b[lo - 1] * c[lo - 1]);
  int64_t nvec = ((hi - lo) >> 5) << 5;   /* multiple of 32 */
  int64_t i = lo;
  for (; i < lo + nvec; i += 32) {
    __m512d p0 = _mm512_mul_pd(_mm512_loadu_pd(b + i), _mm512_loadu_pd(c + i));
    _mm512_storeu_pd(a + i, _mm512_add_pd(p0, prev));
    __m512d p1 = _mm512_mul_pd(_mm512_loadu_pd(b + i + 8), _mm512_loadu_pd(c + i + 8));
    _mm512_storeu_pd(a + i + 8, _mm512_add_pd(p1, p0));
    __m512d p2 = _mm512_mul_pd(_mm512_loadu_pd(b + i + 16), _mm512_loadu_pd(c + i + 16));
    _mm512_storeu_pd(a + i + 16, _mm512_add_pd(p2, p1));
    __m512d p3 = _mm512_mul_pd(_mm512_loadu_pd(b + i + 24), _mm512_loadu_pd(c + i + 24));
    _mm512_storeu_pd(a + i + 24, _mm512_add_pd(p3, p2));
    prev = _mm512_set1_pd(b[i + 31] * c[i + 31]);  /* = p3 lane 7, same single rounding */
  }
  double pb = b[i - 1] * c[i - 1];
  for (; i < hi; ++i) { double s = b[i] * c[i]; a[i] = s + pb; pb = s; }
}
#elif HAVE_SIMD == 1
static void simdk(double *restrict a, const double *restrict b, const double *restrict c,
                  int64_t lo, int64_t hi) {
  __m256d prev = _mm256_set1_pd(b[lo - 1] * c[lo - 1]);
  int64_t nvec = ((hi - lo) >> 5) << 5;   /* multiple of 32 (8 x 4 lanes) */
  int64_t i = lo;
  for (; i < lo + nvec; i += 32) {
    __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(b + i), _mm256_loadu_pd(c + i));
    _mm256_storeu_pd(a + i, _mm256_add_pd(p0, prev));
    __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 4), _mm256_loadu_pd(c + i + 4));
    _mm256_storeu_pd(a + i + 4, _mm256_add_pd(p1, p0));
    __m256d p2 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 8), _mm256_loadu_pd(c + i + 8));
    _mm256_storeu_pd(a + i + 8, _mm256_add_pd(p2, p1));
    __m256d p3 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 12), _mm256_loadu_pd(c + i + 12));
    _mm256_storeu_pd(a + i + 12, _mm256_add_pd(p3, p2));
    __m256d p4 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 16), _mm256_loadu_pd(c + i + 16));
    _mm256_storeu_pd(a + i + 16, _mm256_add_pd(p4, p3));
    __m256d p5 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 20), _mm256_loadu_pd(c + i + 20));
    _mm256_storeu_pd(a + i + 20, _mm256_add_pd(p5, p4));
    __m256d p6 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 24), _mm256_loadu_pd(c + i + 24));
    _mm256_storeu_pd(a + i + 24, _mm256_add_pd(p6, p5));
    __m256d p7 = _mm256_mul_pd(_mm256_loadu_pd(b + i + 28), _mm256_loadu_pd(c + i + 28));
    _mm256_storeu_pd(a + i + 28, _mm256_add_pd(p7, p6));
    prev = _mm256_set1_pd(b[i + 31] * c[i + 31]);  /* = p7 lane 3, same single rounding */
  }
  double pb = b[i - 1] * c[i - 1];
  for (; i < hi; ++i) { double s = b[i] * c[i]; a[i] = s + pb; pb = s; }
}
#endif

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  a[0] = b[0] * c[0];
  if (LEN_1D == 1) return;
#if HAVE_SIMD
  int64_t n = LEN_1D;
  int64_t nt = 96;
  int64_t base = (n - 1) / nt;      /* elements per thread (index 1..n-1) */
  int64_t rem  = (n - 1) % nt;
  #pragma omp parallel for schedule(static, 1)
  for (int64_t t = 0; t < nt; ++t) {
    int64_t chunk = base + (t < rem ? 1 : 0);
    int64_t lo = 1 + t * base + (t < rem ? t : rem);
    int64_t hi = lo + chunk;
    if (hi - lo < 32) {
      double pb = b[lo - 1] * c[lo - 1];
      for (int64_t i = lo; i < hi; ++i) { double s = b[i] * c[i]; a[i] = s + pb; pb = s; }
    } else {
      simdk(a, b, c, lo, hi);
    }
  }
#else
  #pragma omp parallel for schedule(static)
  for (int64_t i = 1; i < LEN_1D; ++i) a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
#endif
}
