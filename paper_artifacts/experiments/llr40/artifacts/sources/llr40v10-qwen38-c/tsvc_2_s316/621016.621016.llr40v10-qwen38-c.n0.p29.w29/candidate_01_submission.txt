#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

static double min_reduce_vec(const double* a, int64_t n) {
  __m512d vmin = _mm512_set1_pd(INFINITY);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m512d v = _mm512_loadu_pd(a + i);
    vmin = _mm512_min_pd(vmin, v);
  }
  double tmp[8];
  _mm512_storeu_pd(tmp, vmin);
  double r = tmp[0];
  for (int j = 1; j < 8; ++j) if (tmp[j] < r) r = tmp[j];
  for (; i < n; ++i) if (a[i] < r) r = a[i];
  return r;
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
  if (LEN_1D <= (1 << 20)) {
    result[0] = min_reduce_vec(a, LEN_1D);
    return;
  }
  double x = a[0];
  #pragma omp parallel reduction(min:x)
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t n = LEN_1D;
    int64_t per = (n + nt - 1) / nt;
    int64_t lo = (int64_t)tid * per;
    int64_t hi = lo + per;
    if (hi > n) hi = n;
    double partial = min_reduce_vec(a + lo, hi - lo);
    if (partial < x) x = partial;
  }
  result[0] = x;
}
