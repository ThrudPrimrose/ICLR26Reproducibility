#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC s2710, fp64.
 *   if (a[i] > b[i]) { a[i] += b[i]*d[i];  LEN>10 ? c[i]+=d[i]*d[i] : c[i]=d[i]*e[i]+1.0; }
 *   else             { b[i]  = a[i]+e[i]*e[i]; x[0]>0 ? c[i]=a[i]+d[i]*d[i] : c[i]+=e[i]*e[i]; }
 * Element-independent: AVX-512 mask arithmetic, 48 threads, contiguous aligned chunks.
 * c is always fully written -> non-temporal full-line stores (32B-aligned), sfenced.
 * d/e are streamed once -> non-temporal loads when 32B-aligned (mode 2).
 */

static inline __m512d loadnt512_pd(const double *restrict q) {
  void *v = (void *)(uintptr_t)q;
  return _mm512_castsi512_pd(_mm512_stream_load_si512(v));
}

#define LD_NORM(p) _mm512_loadu_pd(p)
#define LD_NT(p)   loadnt512_pd(p)

#define STC_NORM \
  _mm512_storeu_pd(c + i, vc)
#define STC_NT \
  do { __m512i vci = _mm512_castpd_si512(vc); \
       _mm256_stream_pd(c + i,     _mm256_castsi256_pd(_mm512_castsi512_si256(vci))); \
       _mm256_stream_pd(c + i + 4, _mm256_castsi256_pd(_mm512_extracti64x4_epi64(vci, 1))); } while (0)

/* PRE/POST split: PRE must run on OLD va; POST consumes the precomputed temps. */
#define PRE_XPOS \
  __m512d badd = _mm512_add_pd(va, ee); \
  __m512d cadd = _mm512_add_pd(va, dd); \
  __m512d bd = _mm512_mul_pd(vb, vd);
#define POST_XPOS \
  vc = _mm512_mask_blend_pd(m, cadd, _mm512_add_pd(vc, dd));
#define PRE_XNEG \
  __m512d badd = _mm512_add_pd(va, ee); \
  __m512d bd = _mm512_mul_pd(vb, vd);
#define POST_XNEG \
  vc = _mm512_add_pd(vc, _mm512_mask_blend_pd(m, ee, dd));

#define DEFINE_BODY(NAME, PRE, POST, LD, ST) \
static inline void NAME(double *restrict a, double *restrict b, double *restrict c, \
                        const double *restrict d, const double *restrict e, int64_t i) { \
  __m512d va = _mm512_loadu_pd(a + i); \
  __m512d vb = _mm512_loadu_pd(b + i); \
  __m512d vc = _mm512_loadu_pd(c + i); \
  __m512d vd = LD(d + i); \
  __m512d ve = LD(e + i); \
  __mmask16 m = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ); \
  __m512d ee = _mm512_mul_pd(ve, ve); \
  __m512d dd = _mm512_mul_pd(vd, vd); \
  PRE \
  va = _mm512_mask_add_pd(va, m, va, bd); \
  vb = _mm512_mask_mov_pd(vb, m ^ 0xFFFFu, badd); \
  POST \
  _mm512_storeu_pd(a + i, va); \
  _mm512_storeu_pd(b + i, vb); \
  ST; \
}

DEFINE_BODY(body_p_0, PRE_XPOS, POST_XPOS, LD_NORM, STC_NORM)
DEFINE_BODY(body_p_1, PRE_XPOS, POST_XPOS, LD_NORM, STC_NT)
DEFINE_BODY(body_p_2, PRE_XPOS, POST_XPOS, LD_NT,   STC_NT)
DEFINE_BODY(body_n_0, PRE_XNEG, POST_XNEG, LD_NORM, STC_NORM)
DEFINE_BODY(body_n_1, PRE_XNEG, POST_XNEG, LD_NORM, STC_NT)
DEFINE_BODY(body_n_2, PRE_XNEG, POST_XNEG, LD_NT,   STC_NT)

static inline void run_xp(double *restrict a, double *restrict b, double *restrict c,
                          const double *restrict d, const double *restrict e,
                          int64_t lo, int64_t hi, int mode) {
  if (mode == 2)      for (int64_t i = lo; i < hi; i += 8) body_p_2(a, b, c, d, e, i);
  else if (mode == 1) for (int64_t i = lo; i < hi; i += 8) body_p_1(a, b, c, d, e, i);
  else                for (int64_t i = lo; i < hi; i += 8) body_p_0(a, b, c, d, e, i);
}

static inline void run_xn(double *restrict a, double *restrict b, double *restrict c,
                          const double *restrict d, const double *restrict e,
                          int64_t lo, int64_t hi, int mode) {
  if (mode == 2)      for (int64_t i = lo; i < hi; i += 8) body_n_2(a, b, c, d, e, i);
  else if (mode == 1) for (int64_t i = lo; i < hi; i += 8) body_n_1(a, b, c, d, e, i);
  else                for (int64_t i = lo; i < hi; i += 8) body_n_0(a, b, c, d, e, i);
}

static inline void tail_scalar(double *restrict a, double *restrict b, double *restrict c,
                               const double *restrict d, const double *restrict e,
                               int xpos, int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      c[i] += d[i] * d[i];
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (xpos) c[i] = a[i] + d[i] * d[i];
      else c[i] += e[i] * e[i];
    }
  }
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  if (LEN_1D <= 10) {
    const int xpos = x[0] > 0.0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
      if (a[i] > b[i]) {
        a[i] += b[i] * d[i];
        c[i] = d[i] * e[i] + 1.0;
      } else {
        b[i] = a[i] + e[i] * e[i];
        if (xpos) c[i] = a[i] + d[i] * d[i];
        else c[i] += e[i] * e[i];
      }
    }
    return;
  }
  const int xpos = x[0] > 0.0;
  const int64_t n = LEN_1D;
  const int64_t nvec = n & ~7LL;
  if (n < 256) {
    for (int64_t i = 0; i < n; ++i) {
      if (a[i] > b[i]) {
        a[i] += b[i] * d[i];
        c[i] += d[i] * d[i];
      } else {
        b[i] = a[i] + e[i] * e[i];
        if (xpos) c[i] = a[i] + d[i] * d[i];
        else c[i] += e[i] * e[i];
      }
    }
    return;
  }
  /* NT stores need c 32B-aligned; NT loads for d/e need them 32B-aligned too
   * (chunk offsets are multiples of 64B, so base alignment suffices). */
  const int mode = ((((uintptr_t)c | (uintptr_t)d | (uintptr_t)e) & 31) == 0) ? 2
                   : ((((uintptr_t)c & 31) == 0) ? 1 : 0);
  #pragma omp parallel num_threads(48)
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t chunk = (nvec / (nt * 8)) * 8;
    if (chunk < 8) {
      /* nvec < nt*8: one element each, scalar */
      if ((int64_t)tid < nvec) {
        const int64_t i = (int64_t)tid;
        if (a[i] > b[i]) { a[i] += b[i] * d[i]; c[i] += d[i] * d[i]; }
        else { b[i] = a[i] + e[i] * e[i]; if (xpos) c[i] = a[i] + d[i] * d[i]; else c[i] += e[i] * e[i]; }
      }
      if (tid == 0) tail_scalar(a, b, c, d, e, xpos, nvec, n);
    } else {
      const int64_t lo = (int64_t)tid * chunk;
      const int64_t hi = (tid == nt - 1) ? nvec : lo + chunk;
      if (xpos) run_xp(a, b, c, d, e, lo, hi, mode);
      else      run_xn(a, b, c, d, e, lo, hi, mode);
      if (tid == nt - 1) tail_scalar(a, b, c, d, e, xpos, nvec, n);
    }
    /* make NT stores to c globally visible before other threads read it */
    if (mode) _mm_sfence();
  }
}
