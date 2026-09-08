#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
  if (LEN_1D <= 1) {
    result[0] = fabs(a[0]);
    return;
  }
  int nt = (LEN_1D >= (1 << 20)) ? omp_get_max_threads() : 1;
  if (nt < 1) nt = 1;
  if (nt > 256) nt = 256;
  static double tm[256];
  static int64_t tx[256];
  const int use_simd = (inc == 1);

  #pragma omp parallel num_threads(nt)
  {
    const int t = omp_get_thread_num();
    const int nt2 = omp_get_num_threads();
    const int64_t per = LEN_1D / nt2;
    const int64_t s = (int64_t)t * per;
    const int64_t e = s + per + (t == nt2 - 1 ? (LEN_1D - s - per) : 0);
    double tb = 0.0;
    int64_t ti = s;
    if (use_simd) {
      const int64_t vs = (s + 7) & ~(int64_t)7;
      const int64_t ve = e & ~(int64_t)7;
      for (int64_t i = s; i < vs; ++i) {
        double v = fabs(a[i]);
        if (v > tb) { tb = v; ti = i; }
      }
      const double *p = a + vs;
      const int64_t nv = ve - vs;
      if (((uintptr_t)p) % 64u == 0u) {
        for (int64_t i = 0; i < nv; i += 8) {
          __m512d x = _mm512_castsi512_pd(_mm512_stream_load_si512((void *)(p + i)));
          x = _mm512_abs_pd(x);
          double cur = _mm512_reduce_max_pd(x);
          if (cur > tb) {
            tb = cur;
            __mmask8 mk = _mm512_cmp_pd_mask(x, _mm512_set1_pd(cur), _CMP_EQ_OQ);
            ti = vs + i + (int64_t)(__builtin_ctz((int)mk));
          }
        }
      } else {
        for (int64_t i = 0; i < nv; i += 8) {
          __m512d x = _mm512_loadu_pd(p + i);
          x = _mm512_abs_pd(x);
          double cur = _mm512_reduce_max_pd(x);
          if (cur > tb) {
            tb = cur;
            __mmask8 mk = _mm512_cmp_pd_mask(x, _mm512_set1_pd(cur), _CMP_EQ_OQ);
            ti = vs + i + (int64_t)(__builtin_ctz((int)mk));
          }
        }
      }
      for (int64_t i = ve; i < e; ++i) {
        double v = fabs(a[i]);
        if (v > tb) { tb = v; ti = i; }
      }
    } else {
      int64_t k = s * inc;
      for (int64_t i = s; i < e; ++i) {
        double v = fabs(a[k]);
        if (v > tb) { tb = v; ti = i; }
        k += inc;
      }
    }
    tm[t] = tb;
    tx[t] = ti;
  }

  double best = tm[0];
  int64_t idx = tx[0];
  for (int t = 1; t < nt; ++t) {
    if (tm[t] > best) { best = tm[t]; idx = tx[t]; }
    else if (tm[t] == best && tx[t] < idx) { idx = tx[t]; }
  }
  result[0] = best + (double)idx;
}
