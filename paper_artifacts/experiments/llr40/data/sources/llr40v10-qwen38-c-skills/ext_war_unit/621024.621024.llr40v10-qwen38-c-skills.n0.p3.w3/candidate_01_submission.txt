/* TSVC tsvc_2_5 / s121:  a[i] = a[i+1] + b[i]  for i in [0, LEN_1D-2].
 *
 * a[i+1] is read by iteration i and written by iteration i+1: a pure
 * anti-dependence, no true recurrence. The ONLY cross-thread conflict is at
 * thread boundaries: thread T's last output reads a[E_T] (old value) while
 * thread T+1's first output writes a[E_T].
 *
 * Protocol: every thread T (except the last) hoists that single boundary
 * load into a register in phase 1; one barrier; phase 2 then runs as an
 * independent, fully vectorized single pass per thread. Each 8-output chunk
 * loads a[i+1..i+8] and b[i..i+7], adds, stores a[i..i+7] -- the reference's
 * 24B per output, no extra passes. */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* 8 outputs a[i..i+7]; reads a[i+1..i+8] and b[i..i+7]. */
static inline void euu_chunk8(double *restrict a, const double *restrict b, int64_t i) {
    __m512d va = _mm512_loadu_pd(a + i + 1);
    __m512d vb = _mm512_loadu_pd(b + i);
    _mm512_storeu_pd(a + i, _mm512_add_pd(va, vb));
}

void ext_war_unit_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, uint8_t *restrict workspace,
                       const int64_t workspace_size) {
  (void)workspace; (void)workspace_size;
  const int64_t n = LEN_1D;
  if (n <= 1) return;
  const int64_t nout = n - 1;                 /* outputs [0, nout) */

  int P = omp_get_max_threads();
  if (P < 1) P = 1;
  if (P > nout) P = (int)nout;

  const int64_t base = nout / P;
  const int64_t rem  = nout % P;

  #pragma omp parallel num_threads(P)
  {
    const int T = omp_get_thread_num();
    const int64_t S = (int64_t)T * base + (T < rem ? T : rem);
    const int64_t E = S + base + (T < rem ? 1 : 0);
    const int Tlast = P - 1;

    double pe = 0.0;
    if (T < Tlast) pe = a[E];                 /* phase 1: boundary load */
    #pragma omp barrier
    /* phase 2 */
    const int64_t vec_end = (T == Tlast) ? nout : E - 1;
    int64_t i = S;
    for (; i + 32 <= vec_end; i += 32) {
      euu_chunk8(a, b, i);
      euu_chunk8(a, b, i + 8);
      euu_chunk8(a, b, i + 16);
      euu_chunk8(a, b, i + 24);
    }
    for (; i + 8 <= vec_end; i += 8) euu_chunk8(a, b, i);
    for (; i < vec_end; ++i) a[i] = a[i + 1] + b[i];
    if (T < Tlast) a[E - 1] = pe + b[E - 1];
  }
}
