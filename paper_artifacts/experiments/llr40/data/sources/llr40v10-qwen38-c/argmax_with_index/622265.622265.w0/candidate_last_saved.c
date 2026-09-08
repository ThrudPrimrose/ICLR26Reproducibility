#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

/* One 8-double chunk: keep (x, idx) where x is the running max of [base, i)
 * and idx its first position. Vector lanes beyond the tail are filled with
 * -INFINITY so they can never win a GT comparison. */
static __attribute__((always_inline)) inline void
amx_vec8(const double *restrict p, int mmask, double *x, int64_t *idx, int64_t base) {
  const __m512d ninf = _mm512_set1_pd(-INFINITY);
  const __m512d v = _mm512_mask_loadu_pd(ninf, (__mmask8)mmask, p);
  const __mmask8 m = _mm512_cmp_pd_mask(v, _mm512_set1_pd(*x), _CMP_GT_OQ);
  if (m) {
    const __m512d vm = _mm512_mask_max_pd(ninf, m, ninf, v);
    const double xv = _mm512_reduce_max_pd(vm);
    const __mmask8 e = _mm512_cmp_pd_mask(v, _mm512_set1_pd(xv), _CMP_EQ_OQ);
    *x = xv;
    *idx = base + (int64_t)__builtin_ctzll((unsigned long long)e);
  }
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 0) {
    *out_value = 0.0;
    *out_index = 0;
    return;
  }
  if (n < (1 << 16)) {
    double xv = a[0];
    int64_t ix = 0;
    for (int64_t i = 1; i < n; ++i)
      if (a[i] > xv) { xv = a[i]; ix = i; }
    *out_value = xv;
    *out_index = ix;
    return;
  }
  int nt = omp_get_max_threads();
  if (nt > 32) nt = 32;
  if (nt < 1) nt = 1;

  double best_v = -INFINITY;
  int64_t best_i = INT64_MAX;
#pragma omp parallel num_threads(nt) reduction(max:best_v) reduction(min:best_i)
  {
    const int64_t chunk = (n + nt - 1) / nt;
    const int64_t s = (int64_t)omp_get_thread_num() * chunk;
    int64_t e = s + chunk;
    if (e > n) e = n;
    double xv = -INFINITY;
    int64_t ix = s;
    int64_t i = s;
    for (; i + 8 <= e; i += 8)
      amx_vec8(a + i, 0xFF, &xv, &ix, i);
    if (i < e)
      amx_vec8(a + i, (1 << (int)(e - i)) - 1, &xv, &ix, i);
    best_v = xv;
    best_i = ix;
  }
  *out_value = best_v;
  *out_index = best_i;
}
