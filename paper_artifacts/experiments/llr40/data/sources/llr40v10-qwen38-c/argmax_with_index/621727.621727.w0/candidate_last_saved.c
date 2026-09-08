/* Optimized argmax_with_index (TSVC s315): running maximum carrying value AND index,
 * first occurrence wins on ties (strict >).
 *
 * Strategy: OpenMP static partition into contiguous chunks; each thread does a
 * single-pass AVX-512 (fallback AVX2 / scalar) scan tracking (max, first index);
 * ordered merge of partials (strict > keeps the earliest on ties).
 *
 * NaN semantics match the scalar reference: a NaN never updates the running max;
 * only a NaN at a[0] sticks forever -> handled by an explicit pre-check.
 * The reported value is the bit pattern of the FIRST occurrence of the maximum
 * (matters only for -0.0 vs +0.0; a volatile load keeps it exact under
 * -fno-signed-zeros).
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define VECTOR_PATH 2 /* AVX512 */
#elif defined(__AVX2__)
#include <immintrin.h>
#define VECTOR_PATH 1 /* AVX2 */
#else
#define VECTOR_PATH 0 /* scalar */
#endif

typedef struct {
  double v;
  int64_t i;
} amx_t;

#if VECTOR_PATH == 2
static inline double hmax512(__m512d v) {
  const __m512i p4 = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
  const __m512i p2 = _mm512_setr_epi64(2, 3, 4, 5, 6, 7, 0, 1);
  const __m512i p1 = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 0);
  __m512d t = v;
  t = _mm512_max_pd(t, _mm512_permutexvar_pd(p4, t));
  t = _mm512_max_pd(t, _mm512_permutexvar_pd(p2, t));
  t = _mm512_max_pd(t, _mm512_permutexvar_pd(p1, t));
  double tmp[8];
  _mm512_storeu_pd(tmp, t);
  return tmp[0];
}
#endif

#if VECTOR_PATH == 1
static inline double hmax256(__m256d v) {
  __m256d t1 = _mm256_max_pd(v, _mm256_shuffle_pd(v, v, 5));
  __m256d t2 = _mm256_max_pd(t1, _mm256_shuffle_pd(t1, t1, 1));
  return _mm_cvtsd_f64(_mm256_extractf128_pd(t2, 0));
}
#endif

/* Single-pass argmax over a[lo, hi). Caller guarantees whole-array a[0] is not NaN
 * (sticky-NaN case handled there), so a chunk scan starting from -inf matches the
 * reference semantics exactly. */
static amx_t chunk_amax(const double *restrict a, int64_t lo, int64_t hi, int use_nt) {
  amx_t r;
  double best = -INFINITY;
  int64_t bi = lo;
  int64_t i = lo;
#if VECTOR_PATH == 0
  (void)use_nt;
#endif

#if VECTOR_PATH == 2
#define AMX512_STEP(_v)                                                                  \
  do {                                                                                   \
    __mmask8 _m = _mm512_cmp_pd_mask(_v, bset, _CMP_GT_OQ);                              \
    if (_m) {                                                                            \
      __m512d _vv = _mm512_mask_mov_pd(bset, _m, _v);                                    \
      best = hmax512(_vv);                                                               \
      __mmask8 _e = _mm512_cmp_pd_mask(_v, _mm512_set1_pd(best), _CMP_EQ_OQ);            \
      int _lane = __builtin_ctz((int)_e);                                                \
      double _tmp[8];                                                                    \
      _mm512_storeu_pd(_tmp, _v);                                                        \
      best = ((const volatile double *)_tmp)[_lane];                                     \
      bset = _mm512_set1_pd(best);                                                       \
      bi = (i) + (int64_t)_lane;                                                         \
    }                                                                                    \
  } while (0)
  if (use_nt) {
    /* scalar-advance to a 64-byte aligned element (<= 7 iters), then stream-load */
    for (; i < hi && ((uintptr_t)(a + i) & 63u); ++i) {
      double d = a[i];
      if (d > best) {
        best = d;
        bi = i;
      }
    }
    __m512d bset = _mm512_set1_pd(best);
    for (; i + 8 <= hi; i += 8) {
      __m512d v = _mm512_castsi512_pd(_mm512_stream_load_si512((void *)(a + i)));
      AMX512_STEP(v);
    }
  } else {
    __m512d bset = _mm512_set1_pd(best);
    for (; i + 8 <= hi; i += 8) {
      __m512d v = _mm512_loadu_pd(a + i);
      AMX512_STEP(v);
    }
  }
#undef AMX512_STEP
#elif VECTOR_PATH == 1
  if (use_nt) {
    for (; i < hi && ((uintptr_t)(a + i) & 31u); ++i) {
      double d = a[i];
      if (d > best) {
        best = d;
        bi = i;
      }
    }
  }
  for (; i + 4 <= hi; i += 4) {
    __m256d v = use_nt
                    ? _mm256_castsi256_pd(_mm256_stream_load_si256((void *)(a + i)))
                    : _mm256_loadu_pd(a + i);
    __m256d bset = _mm256_set1_pd(best);
    __m256d gtv = _mm256_cmp_pd(v, bset, _CMP_GT_OQ);
    int m = _mm256_movemask_pd(gtv);
    if (m) {
      __m256d vv = _mm256_or_pd(_mm256_and_pd(gtv, v), _mm256_andnot_pd(gtv, bset));
      best = hmax256(vv);
      __m256d eqv = _mm256_cmp_pd(v, _mm256_set1_pd(best), _CMP_EQ_OQ);
      int lane = __builtin_ctz(_mm256_movemask_pd(eqv));
      double tmp[4];
      _mm256_storeu_pd(tmp, v);
      best = ((const volatile double *)tmp)[lane];
      bi = i + (int64_t)lane;
    }
  }
#endif

  for (; i < hi; ++i) {
    double d = a[i];
    if (d > best) {
      best = d;
      bi = i;
    }
  }
  r.v = best;
  r.i = bi;
  return r;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
  if (LEN_1D <= 0)
    return;
  /* Sticky-NaN case: a[0] NaN => reference never updates: (a[0], 0). */
  if (isnan(a[0])) {
    out_value[0] = a[0];
    out_index[0] = 0;
    return;
  }
  if (LEN_1D == 1) {
    out_value[0] = a[0];
    out_index[0] = 0;
    return;
  }

  int maxt = omp_get_max_threads();
  int nt = (int)((LEN_1D + 65535) / 65536);
  if (nt > maxt)
    nt = maxt;
  if (nt < 1)
    nt = 1;

  amx_t partial[128];
  if (nt > 128)
    nt = 128;

  #pragma omp parallel num_threads(nt)
  {
    int tid = omp_get_thread_num();
    int ntc = omp_get_num_threads();
    int64_t base = LEN_1D / ntc;
    int64_t rem = LEN_1D % ntc;
    int64_t lo = (int64_t)tid * base + (tid < rem ? tid : rem);
    int64_t hi = lo + base + (tid < rem ? 1 : 0);
    int use_nt = 0;
#if VECTOR_PATH != 0
    if (hi - lo >= (1 << 27))
      use_nt = 1; /* 256 MB per chunk: stream it */
#endif
    amx_t r = chunk_amax(a, lo, hi, use_nt);
    partial[tid] = r;
  }

  amx_t acc = partial[0];
  for (int t = 1; t < nt; ++t) {
    if (partial[t].v > acc.v)
      acc = partial[t];
  }
  out_value[0] = acc.v;
  out_index[0] = acc.i;
}
