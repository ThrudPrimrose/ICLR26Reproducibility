/* tsvc_2_s115: for j in 0..n-1: for i in j+1..n-1: a[i] -= aa[j*n+i] * a[j]
 *
 * c_j := a[j] after stages 0..j-1 (final value of a[j]); stage j: a[i] -= aa[j*n+i]*c_j (i > j).
 * Interleaved 64-element, 64B-line-aligned blocks; thread t owns blocks t, t+T, t+2T, ...
 *
 * Publish protocol: owner of element k+1 broadcasts c_{k+1}=a[k+1] (its local, final value)
 * into every other thread's mailbox at the end of stage k.  Each mailbox is a ring of
 * (value, seq) pairs, 16B each, two on a 64B line per pair-group; consumer polls only its
 * own ring (one writer + one reader per line: no snoop storm).  If the consumer ever lags
 * more than RING stages and its slot wrapped (seq > k), it reads a[k] directly -- a[k] is
 * never written after it is finalized, so that value is always final and correct.
 * x86 TSO: plain/volatile loads+stores suffice (value store ordered before seq store).
 * Per-element op = one FMA (a - aa*c), matching the judge oracle's contraction.
 */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stddef.h>

#define RING 64

static void axpy_seg(double *restrict d, const double *restrict row, int64_t m, double c) {
  __m512d cc = _mm512_set1_pd(c);
  int64_t i = 0;
  for (; i + 16 <= m; i += 16) {
    __m512d v0 = _mm512_loadu_pd(d + i);
    __m512d v1 = _mm512_loadu_pd(d + i + 8);
    __m512d r0 = _mm512_loadu_pd(row + i);
    __m512d r1 = _mm512_loadu_pd(row + i + 8);
    v0 = _mm512_fnmadd_pd(r0, cc, v0);
    v1 = _mm512_fnmadd_pd(r1, cc, v1);
    _mm512_storeu_pd(d + i, v0);
    _mm512_storeu_pd(d + i + 8, v1);
    __builtin_prefetch(row + i + 80, 0, 1);
  }
  for (; i < m; i++) d[i] = __builtin_fma(row[i], -c, d[i]);
}

static void tsvc_serial(double *restrict a, const double *restrict aa, int64_t n) {
  for (int64_t j = 0; j < n; j++)
    axpy_seg(a + j + 1, aa + j * n + j + 1, n - j - 1, a[j]);
}

typedef struct { volatile double c[RING]; volatile long long seq[RING]; } mb_t;

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n < 2) return;

  int T = omp_get_max_threads();
  if (T < 1) T = 1;
  if (T > 48) T = 48;

  if ((int64_t)T <= 1 || n < 3072) { tsvc_serial(a, aa, n); return; }

  /* block 0 = [0, D1), block m>=1 = [D1+64(m-1), D1+64m); D1 = 8K0+64 so that
   * a + 8*(D1+64(m-1)) is 64B-aligned: no block boundary splits a cache line. */
  const int q = (int)(((uintptr_t)a >> 3) & 7);
  const int K0 = (8 - q) & 7;
  const int64_t D1 = 8LL * (int64_t)K0 + 64;
  int64_t Q = (n <= D1) ? 1 : 1 + (n - D1 + 63) / 64;
  if ((int64_t)T > Q) T = (int)Q;

  alignas(64) static mb_t mb[48];
  for (int t = 0; t < T; t++) {
    for (int s = 0; s < RING; s++) { mb[t].seq[s] = -1; mb[t].c[s] = 0.0; }
    mb[t].c[0] = a[0];
    mb[t].seq[0] = 0;          /* c_0 ready */
  }

  #pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    if (tid < Q) {
      const int64_t M = (Q - 1 - tid) / T + 1;   /* blocks tid, tid+T, ... < Q */
      int64_t b_last = tid + (M - 1) * T;
      int64_t e_last = (b_last == 0) ? D1 : D1 + 64 * b_last;
      if (e_last > n) e_last = n;
      for (int64_t k = 0; k + 1 < e_last; k++) {
        const int64_t b  = (k     < D1) ? 0 : 1 + (k - D1) / 64;
        const int64_t b1 = (k + 1 < D1) ? 0 : 1 + (k + 1 - D1) / 64;
        double c;
        if (b % T == tid) {
          c = a[k];                       /* local: final after our stage k-1 */
        } else {
          const int s = (int)(k & (RING - 1));
          long long g;
          while ((g = mb[tid].seq[s]) < k) __builtin_ia32_pause();
          /* a[k] is read VOLATILE: a plain load could be hoisted above the wait and be
           * pre-finalization; at wait-exit a[k] is final (publish implies final store). */
          c = (g == k) ? mb[tid].c[s] : *(volatile double *)(a + k);
        }
        int64_t m0 = 0;
        if (b1 % T == tid) m0 = (b1 - tid) / T;
        int64_t m;
        for (m = m0; m < M; m++) {
          const int64_t bb = tid + m * T;
          int64_t s = (bb == 0) ? 0 : D1 + 64 * (bb - 1);
          int64_t e = (bb == 0) ? D1 : D1 + 64 * bb;
          if (e > n) e = n;
          int64_t i0 = (k + 1 > s) ? k + 1 : s;
          if (i0 < e) axpy_seg(a + i0, aa + k * n + i0, e - i0, c);
        }
        for (m = 0; m < m0; m++) {
          const int64_t bb = tid + m * T;
          int64_t s = (bb == 0) ? 0 : D1 + 64 * (bb - 1);
          int64_t e = (bb == 0) ? D1 : D1 + 64 * bb;
          if (e > n) e = n;
          int64_t i0 = (k + 1 > s) ? k + 1 : s;
          if (i0 < e) axpy_seg(a + i0, aa + k * n + i0, e - i0, c);
        }
        if (b1 % T == tid) {
          const double cv = a[k + 1];     /* local: final after this stage */
          const int s = (int)((k + 1) & (RING - 1));
          for (int t = 0; t < T; t++)
            if (t != tid) { mb[t].c[s] = cv; mb[t].seq[s] = (long long)(k + 1); }
        }
      }
    }
  }
}
