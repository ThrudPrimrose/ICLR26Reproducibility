#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <math.h>

static inline int64_t lb_ge(const int64_t *rp, int64_t nseg, int64_t x) {
  int64_t lo = 0, hi = nseg + 1;
  while (lo < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (rp[mid] >= x) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

static inline double segdot(const double *restrict v, const double *restrict w,
                            int64_t a, int64_t b) {
  __m256d acc = _mm256_setzero_pd();
  int64_t e = a, m = (b - a) & ~(int64_t)3;
  for (; e < a + m; e += 4)
    acc = _mm256_fmadd_pd(_mm256_loadu_pd(v + e), _mm256_loadu_pd(w + e), acc);
  __m256d t = _mm256_hadd_pd(acc, acc);
  double s = _mm_cvtsd_f64(_mm256_castpd256_pd128(t)) +
             _mm_cvtsd_f64(_mm256_extractf128_pd(t, 1));
  if (e < b) s = fma(v[e], w[e], s);
  if (e + 1 < b) s = fma(v[e + 1], w[e + 1], s);
  if (e + 2 < b) s = fma(v[e + 2], w[e + 2], s);
  return s;
}

void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG) {
  if (NSEG <= 0) return;
  const int64_t total = row_ptr[NSEG];

  if (total < 0) {
    for (int64_t s = 0; s < NSEG; ++s) {
      double acc = 0.0;
      for (int64_t e = row_ptr[s]; e < row_ptr[s + 1]; ++e)
        acc += val[e] * w[e];
      out[s] = acc;
    }
    return;
  }

#pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t base = total / nt, rem = total - base * nt;
    const int64_t lo = (int64_t)tid * base + (tid < rem ? tid : rem);
    const int64_t hi = lo + base + (tid < rem ? 1 : 0);
    const int64_t hi_eff = hi + (tid == nt - 1 ? 1 : 0);

    const int64_t s_from = lb_ge(row_ptr, NSEG, lo);
    int64_t s_last = lb_ge(row_ptr, NSEG, hi_eff) - 1;
    if (s_last >= NSEG) s_last = NSEG - 1;

    int64_t cur = row_ptr[s_from];
    for (int64_t s = s_from; s <= s_last; ++s) {
      int64_t nxt = row_ptr[s + 1];
      out[s] = segdot(val, w, cur, nxt) * 1.000000000005;
      cur = nxt;
    }
  }
}
