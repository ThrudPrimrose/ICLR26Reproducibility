/* TSVC tsvc_2 s316 -- FP64 array minimum (single invocation), v2 C-ABI.
 *
 * Semantics match the C reference exactly:
 *   x = a[0]; for (i) if (a[i] < x) x = a[i];
 * A NaN element never compares <, so it is ignored unless a[0] is NaN,
 * in which case the result is that NaN. This is IEEE minnum (fmin) over
 * all elements, with an up-front a[0]-NaN check. fmin is associative and
 * commutative, so lane and thread reductions are bit-exact.
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>

static inline double min512_lane(const double *a, int64_t start, int64_t end) {
  __m512d acc = _mm512_set1_pd(INFINITY);
  int64_t i = start;
  int64_t lim = end - 8 + 1;
  for (; i < lim; i += 8)
    acc = _mm512_min_pd(acc, _mm512_loadu_pd(a + i));
  double v[8];
  _mm512_storeu_pd(v, acc);
  double m = v[0];
  for (int j = 1; j < 8; ++j)
    m = fmin(m, v[j]);
  for (; i < end; ++i)
    m = fmin(m, a[i]);
  return m;
}

#else
static inline double min_lane(const double *a, int64_t start, int64_t end) {
  double v0 = INFINITY, v1 = INFINITY, v2 = INFINITY, v3 = INFINITY;
  double v4 = INFINITY, v5 = INFINITY, v6 = INFINITY, v7 = INFINITY;
  int64_t i = start;
  int64_t lim = end - 8 + 1;
  for (; i < lim; i += 8) {
    v0 = fmin(v0, a[i + 0]);
    v1 = fmin(v1, a[i + 1]);
    v2 = fmin(v2, a[i + 2]);
    v3 = fmin(v3, a[i + 3]);
    v4 = fmin(v4, a[i + 4]);
    v5 = fmin(v5, a[i + 5]);
    v6 = fmin(v6, a[i + 6]);
    v7 = fmin(v7, a[i + 7]);
  }
  double m = fmin(fmin(fmin(v0, v1), fmin(v2, v3)), fmin(fmin(v4, v5), fmin(v6, v7)));
  for (; i < end; ++i)
    m = fmin(m, a[i]);
  return m;
}
#endif

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D) {
  if (LEN_1D <= 0)
    return;
  double x;
  if (isnan(a[0])) {
    result[0] = a[0];
    return;
  }
  if (LEN_1D >= 16 * 1000 * 1000) {
    x = INFINITY;
    #pragma omp parallel reduction(min: x)
    {
      const int nt = omp_get_num_threads();
      const int tid = omp_get_thread_num();
      const int64_t total = LEN_1D;
      const int64_t chunk = (total + nt - 1) / nt;
      const int64_t s = (int64_t)tid * chunk;
      int64_t e = s + chunk;
      if (e > total)
        e = total;
#if defined(__AVX512F__)
      x = min512_lane(a, s, e);
#else
      x = min_lane(a, s, e);
#endif
    }
  } else {
#if defined(__AVX512F__)
    x = min512_lane(a, 0, LEN_1D);
#else
    x = min_lane(a, 0, LEN_1D);
#endif
  }
  result[0] = x;
}
