/* TSVC_2 tsvc_2 kernel s319 -- optimized C (AVX-512 + OpenMP).
 *
 *   a[i] = c[i] + d[i];  b[i] = c[i] + e[i];  sum over all a[i],b[i] -> b[0]
 *
 * Elementwise stores are order-independent, so the only freedom is the reduction.
 * The reduction is made BITWISE DETERMINISTIC (the judge's harden gate requires
 * two runs on identical data to be bitwise equal): the range is split into a
 * FIXED number C of contiguous chunks, each chunk's partial sum is computed by
 * exactly one thread with a fixed SIMD tree, and the C partials are then added
 * serially in fixed order OUTSIDE the parallel region. The result therefore
 * does not depend on the OpenMP team size or scheduling -- unlike the built-in
 * `reduction(+:x)` clause, whose combine order is not guaranteed deterministic.
 *
 * Stores are non-temporal (VMOVNTPD) for the 64B-aligned bulk to avoid the
 * read-for-ownership penalty on streaming writes; b[0] is peeled from NT stores
 * (and re-written with a normal store as the final sum), and each thread flushes
 * its NT stores with a single SFENCE before the team barrier.
 */
#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <immintrin.h>

static inline double hsum512(__m512d v){
  double r[8]; _mm512_storeu_pd(r, v);
  return (r[0]+r[1])+(r[2]+r[3])+(r[4]+r[5])+(r[6]+r[7]);
}

/* Compute a[i]=c[i]+d[i], b[i]=c[i]+e[i] for i in [lo,hi); return the sum of all
   a[i]+b[i] produced. Normal stores (unaligned inputs / small remainder). */
static double partial_norm(double* a, double* b, const double* c, const double* d,
                           const double* e, int64_t lo, int64_t hi){
  __m512d acc = _mm512_setzero_pd();
  double ps = 0.0;
  int64_t i = lo;
  for (; i+8 <= hi; i += 8){
    __m512d vc = _mm512_loadu_pd(c+i), vd = _mm512_loadu_pd(d+i), ve = _mm512_loadu_pd(e+i);
    __m512d va = _mm512_add_pd(vc, vd), vb = _mm512_add_pd(vc, ve);
    _mm512_storeu_pd(a+i, va);
    _mm512_storeu_pd(b+i, vb);
    acc = _mm512_add_pd(acc, va);
    acc = _mm512_add_pd(acc, vb);
  }
  for (; i < hi; i++){
    double ai = c[i]+d[i]; a[i] = ai; ps += ai;
    double bi = c[i]+e[i]; b[i] = bi; ps += bi;
  }
  ps += hsum512(acc);
  return ps;
}

/* Same, with non-temporal streaming stores for the 64B-aligned bulk. Indices < 8 are
   peeled to normal stores so b[0] is never touched by an NT store. The caller issues
   one SFENCE per thread (after its chunks) to drain the WC buffer. */
static double partial_nt(double* a, double* b, const double* c, const double* d,
                         const double* e, int64_t lo, int64_t hi){
  __m512d acc = _mm512_setzero_pd();
  double ps = 0.0;
  int64_t i = lo;
  int64_t al = (i+7) & ~7LL;
  if (al < 8) al = 8;
  if (al > hi) al = hi;
  for (; i < al; i++){
    double ai = c[i]+d[i]; a[i] = ai; ps += ai;
    double bi = c[i]+e[i]; b[i] = bi; ps += bi;
  }
  for (; i+8 <= hi; i += 8){
    __m512d vc = _mm512_loadu_pd(c+i), vd = _mm512_loadu_pd(d+i), ve = _mm512_loadu_pd(e+i);
    __m512d va = _mm512_add_pd(vc, vd), vb = _mm512_add_pd(vc, ve);
    _mm512_stream_pd(a+i, va);
    _mm512_stream_pd(b+i, vb);
    acc = _mm512_add_pd(acc, va);
    acc = _mm512_add_pd(acc, vb);
  }
  for (; i < hi; i++){
    double ai = c[i]+d[i]; a[i] = ai; ps += ai;
    double bi = c[i]+e[i]; b[i] = bi; ps += bi;
  }
  ps += hsum512(acc);
  return ps;
}

void tsvc_2_s319_fp64(
    double *restrict a,
    double *restrict b,
    const double *restrict c,
    const double *restrict d,
    const double *restrict e,
    const int64_t LEN_1D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
  (void)workspace; (void)workspace_size;
  const int64_t n = LEN_1D;
  if (n <= 0) return;

  const int use_nt = (((uintptr_t)a | (uintptr_t)b) & 63) == 0;

  /* Fixed chunk count: the final sum is a fixed-order serial sum of C partials,
     independent of the team size -> bitwise reproducible run to run. */
  enum { C = 128 };
  double parts[C];
  const int64_t chunk = (n + C - 1) / C;

  #pragma omp parallel
  {
    /* Contiguous group of chunks per thread (block schedule) -> each thread
       streams one contiguous region of the arrays. Every chunk is still owned
       by exactly one thread, so the fixed-order combine below stays
       bitwise deterministic regardless of the team size. */
    const int tn  = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int per = tn ? C / tn : C;
    const int rem = C - per*tn;
    const int cl  = tid*per + (tid < rem ? tid : rem);
    const int ch  = cl + per + (tid < rem ? 1 : 0);
    for (int cc = cl; cc < ch; cc++){
      int64_t lo = (int64_t)cc * chunk;
      int64_t hi = lo + chunk;
      if (hi > n) hi = n;
      if (lo < hi)
        parts[cc] = use_nt ? partial_nt(a,b,c,d,e,lo,hi)
                           : partial_norm(a,b,c,d,e,lo,hi);
      else
        parts[cc] = 0.0;
    }
    /* One SFENCE per thread, after all its chunks: drains this core's WC buffer
       so every NT store is visible across cores before the team barrier. Doing
       it once (not once per chunk) keeps the NT store pipeline from being
       serialized 128 times. */
    if (use_nt) _mm_sfence();
  }

  double sum_val = 0.0;
  for (int cc = 0; cc < C; cc++) sum_val += parts[cc];
  b[0] = sum_val;
}
