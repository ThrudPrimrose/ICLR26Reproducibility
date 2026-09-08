/* Hand port of the TSVC tsvc_2 microkernel ``s1244`` (s1244_d_single.cpp), fp64
 * single-invocation variant, to C23 under the v2 C-ABI.
 *
 * Semantics (identical to the reference):
 *   for i in [0, LEN_1D-2]:
 *     a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
 *     d[i] = a[i] + a[i+1]
 * Crucially, d[i] reads a[i+1] BEFORE iteration i+1 overwrites it, i.e. it uses the
 * ORIGINAL input value of a[i+1].  a[LEN_1D-1] is never written.
 *
 * Parallelization: each thread gets a static contiguous range [i0, i1).  Iteration i
 * reads a[i+1] and iteration i+1 writes a[i+1] -- an anti-dependence at distance 1.
 * Inside one thread the serial order keeps every a[i+1] read at its original value.
 * The only cross-thread touch is the boundary element a[i1]: thread t reads it (last
 * iteration) while thread t+1 writes it (first iteration).  Protocol: at region entry
 * each thread loads bnd = a[i1] into a scalar; a barrier separates the reads from the
 * streaming writes.  After the barrier every a element is written by exactly one
 * thread and read (as a[i+1]) only by the owning thread.
 *
 * The scalar loop does not auto-vectorize (the a[i+1] load overlaps the a[i] store),
 * so the hot path is hand-vectorized with AVX-512: each vector loads 8 doubles from
 * b, from c and the original a[i+1..i+8], computes the new values and stores a and d.
 * Loads issue before stores, so the overlapped a elements keep their original values.
 */

#include <stdint.h>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static inline void s1244_scalar(double *restrict a, const double *restrict b,
                                const double *restrict c, double *restrict d,
                                int64_t i, double next_a) {
  const double bi = b[i];
  const double ci = c[i];
  const double v  = bi + ci * ci + bi * bi + ci;
  a[i] = v;
  d[i] = v + next_a;
}

#ifdef __AVX512F__
static inline void s1244_vec8(double *restrict a, const double *restrict b,
                              const double *restrict c, double *restrict d,
                              int64_t i) {
  const __m512d vb = _mm512_loadu_pd(b + i);
  const __m512d vc = _mm512_loadu_pd(c + i);
  /* original a[i+1..i+8]: loaded before any store below touches a[i..i+7] */
  const __m512d va = _mm512_loadu_pd(a + i + 1);
  __m512d v = _mm512_fmadd_pd(vc, vc, vb);   /* b + c*c                */
  v = _mm512_fmadd_pd(vb, vb, v);            /* (b + c*c) + b*b        */
  v = _mm512_add_pd(v, vc);                  /* (b + c*c) + b*b + c    */
  _mm512_storeu_pd(a + i, v);
  _mm512_storeu_pd(d + i, _mm512_add_pd(v, va));
}
#endif

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {

  const int64_t N = LEN_1D - 1;
  if (N <= 0) return;

#ifdef __AVX512F__
  if (N >= (1 << 16) && omp_get_max_threads() >= 2) {
    #pragma omp parallel
    {
      const int     nt   = omp_get_num_threads();
      const int     tid  = omp_get_thread_num();
      const int64_t base = N / nt;
      const int64_t rem  = N % nt;
      const int64_t i0   = (int64_t)tid * base + (tid < rem ? tid : rem);
      const int64_t len  = base + (tid < rem);
      const int64_t i1   = i0 + len;
      /* boundary value: original a[i1], needed as a[i+1] for i = i1-1.
       * Read at region entry; no thread has written anything yet. */
      const double bnd = a[i1];
      #pragma omp barrier
      const int64_t iend = i1 - 1;
      int64_t i = i0;
      /* peel so the a[i..i+7] store (and b, c, d) are 64B aligned;
       * a[i+1..i+8] stays 8B off, a single split load on Zen */
      while (i < iend && ((uintptr_t)(a + i) & 63) != 0) {
        s1244_scalar(a, b, c, d, i, a[i + 1]);
        i++;
      }
      for (; i + 8 <= iend; i += 8)
        s1244_vec8(a, b, c, d, i);
      for (; i < iend; i++)
        s1244_scalar(a, b, c, d, i, a[i + 1]);
      if (len > 0)
        s1244_scalar(a, b, c, d, i1 - 1, bnd);
    }
    return;
  }
#endif

  for (int64_t i = 0; i < N; i++)
    s1244_scalar(a, b, c, d, i, a[i + 1]);
}
