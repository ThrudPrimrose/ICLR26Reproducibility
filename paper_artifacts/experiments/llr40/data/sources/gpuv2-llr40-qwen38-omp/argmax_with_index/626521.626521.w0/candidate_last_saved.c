/* TSVC tsvc_2_5 argmax_with_index: running maximum carrying BOTH value and index.
 *
 * Strategy (OpenMP offload arm, MI300A):
 *  - large inputs: map the input to the device once, parallel argmax via
 *    `teams distribute parallel for` with a custom (value,index) reduction.
 *  - tiny inputs: host fast path; the offload round trip is not worth it.
 *  - a[0] == NaN: the reference's running max is poisoned (nothing > NaN),
 *    so the answer is always (NaN, 0).
 */
#include <stdint.h>
#include <math.h>

typedef struct { double v; int64_t i; } amax_t;

static amax_t amax_init(void) {
  amax_t r;
  r.v = -INFINITY;
  r.i = -1;
  return r;
}

static amax_t amax_comb(amax_t a, amax_t b) {
  if (b.v > a.v) return b;
  if (a.v > b.v) return a;
  /* equal (incl. both -INF): keep the earliest index */
  if (a.i < 0) return b;
  if (b.i < 0) return a;
  return (a.i <= b.i) ? a : b;
}
#pragma omp declare reduction (amxr: amax_t : x = amax_comb(x, y)) init(amax_init())

#define AMAX_TP 256    /* threads per team  */
#define AMAX_NT 1024   /* teams             */
#define AMAX_HOST_CUTOFF (1 << 20)

static void argmax_host(const double *restrict a, int64_t *restrict out_index,
                        double *restrict out_value, const int64_t n) {
  double x = a[0];
  int64_t idx = 0;
  for (int64_t i = 1; i < n; ++i) {
    if (a[i] > x) {
      x = a[i];
      idx = i;
    }
  }
  out_value[0] = x;
  out_index[0] = idx;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  if (LEN_1D == 1) {
    out_value[0] = a[0];
    out_index[0] = 0;
    return;
  }
  if (isnan(a[0])) {
    out_value[0] = a[0];
    out_index[0] = 0;
    return;
  }
  if (LEN_1D < AMAX_HOST_CUTOFF) {
    argmax_host(a, out_index, out_value, LEN_1D);
    return;
  }
  amax_t res;
  #pragma omp target map(to: a[0:LEN_1D]) map(tofrom: res)
  {
    #pragma omp teams distribute parallel for num_teams(AMAX_NT) num_threads(AMAX_TP) reduction(amxr: res)
    for (int64_t i = 0; i < LEN_1D; ++i) {
      if (a[i] > res.v) {
        res.v = a[i];
        res.i = i;
      }
    }
  }
  out_value[0] = res.v;
  out_index[0] = res.i;
}
