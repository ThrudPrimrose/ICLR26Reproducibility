#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

/* argmax (first occurrence) over fp64.
 *
 * A (value, index) pair with "max value wins, on ties the smaller index wins"
 * is a commutative + associative semigroup, so any partition of the array into
 * chunks, a first-occurrence scan per chunk, and an arbitrary fold of the chunk
 * results reproduces the reference exactly.
 */
typedef struct { double v; int64_t i; } amx_t;

static inline amx_t amx_id(void) {
  amx_t r = { -INFINITY, INT64_MAX };
  return r;
}

static inline amx_t amx_combine(amx_t a, amx_t b) {
  if (b.v > a.v || (a.v == b.v && b.i < a.i)) return b;
  return a;
}

/* scalar first-occurrence scan (device-safe: no x86 intrinsics) */
static inline amx_t amx_scan_scalar(const double *a, int64_t b, int64_t e) {
  amx_t acc = amx_id();
  for (int64_t i = b; i < e; ++i)
    if (a[i] > acc.v) { acc.v = a[i]; acc.i = i; }
  return acc;
}

/* single-pass AVX-512 first-occurrence scan for the host */
static inline amx_t amx_scan_host(const double *a, int64_t b, int64_t e) {
  double bv = -INFINITY;
  int64_t bi = INT64_MAX;
  int64_t i = b;
  const int64_t vstop = e - 7;
  /* peel to a 64-byte boundary: loadu is one uop only when aligned */
  for (; i < vstop && (i & 7) != 0; ++i)
    if (a[i] > bv) { bv = a[i]; bi = i; }
  for (; i < vstop; i += 8) {
    const __m512d v = _mm512_loadu_pd(a + i);
    const double vm = _mm512_reduce_max_pd(v);
    if (vm > bv) {
      const __mmask8 m = _mm512_cmpeq_pd_mask(v, _mm512_set1_pd(vm));
      bi = i + __builtin_ctz(m);
      bv = vm;
    }
  }
  for (; i < e; ++i)
    if (a[i] > bv) { bv = a[i]; bi = i; }
  return (amx_t) { bv, bi };
}

#define SERIAL_MAX   (1 << 18)    /* below: one vectorized serial pass (no fork) */
#define DEVICE_MIN_N (1 << 22)    /* below: skip the offloaded pass entirely */
#define DEVICE_CHUNK (1 << 19)    /* elements the device pass covers */
#define DEVT         64           /* fixed device geometry */
#define DEVTSL       128
#define MAXSLOT      (DEVT * DEVTSL)

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 1) {
    out_value[0] = (n > 0) ? a[0] : 0.0;
    out_index[0] = 0;
    return;
  }
  if (n <= SERIAL_MAX) {
    amx_t r = amx_scan_host(a, 0, n);
    out_value[0] = r.v;
    out_index[0] = r.i;
    return;
  }

  int64_t dchunk = 0;
  amx_t r = amx_id();
  if (n >= DEVICE_MIN_N) {
    dchunk = DEVICE_CHUNK;
    amx_t *dp = (amx_t *)malloc((size_t)MAXSLOT * sizeof(amx_t));
    int64_t pcount = 0;
    #pragma omp target map(to : a[0:dchunk]) map(tofrom : dp[0:MAXSLOT]) map(tofrom : pcount)
    {
      #pragma omp teams num_teams(DEVT) thread_limit(DEVTSL)
      {
        #pragma omp parallel
        {
          const int64_t nt = omp_get_num_teams();
          const int64_t ts = omp_get_max_threads();
          const int64_t P = nt * ts;
          const int64_t g = (int64_t)omp_get_team_num() * ts + omp_get_thread_num();
          if (g < P) {
            const int64_t span = (dchunk + P - 1) / P;
            const int64_t b = g * span;
            int64_t e = b + span;
            if (e > dchunk) e = dchunk;
            dp[g] = amx_scan_scalar(a, b, e);
          }
          if (g == 0) pcount = P;
        }
      }
    }
    for (int64_t t = 0; t < pcount; ++t) r = amx_combine(r, dp[t]);
    free(dp);
  }

  const int T = omp_get_max_threads();
  amx_t *pp = (amx_t *)malloc((size_t)T * sizeof(amx_t));
  #pragma omp parallel shared(pp, a)
  {
    const int64_t T2 = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t lo = dchunk, hi = n;
    const int64_t span = (hi - lo + T2 - 1) / T2;
    const int64_t b = lo + (int64_t)tid * span;
    int64_t e = b + span;
    if (e > hi) e = hi;
    pp[tid] = amx_scan_host(a, b, e);
  }
  for (int t = 0; t < T; ++t) r = amx_combine(r, pp[t]);
  free(pp);

  out_value[0] = r.v;
  out_index[0] = r.i;
}
