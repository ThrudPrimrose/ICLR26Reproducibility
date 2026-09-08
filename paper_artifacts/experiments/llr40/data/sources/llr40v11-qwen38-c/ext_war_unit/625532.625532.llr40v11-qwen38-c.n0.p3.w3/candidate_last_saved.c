#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* a[i] = A[i+1] + b[i]  (i in [0,N-2]); a[N-1] unchanged.
 * Pointwise with shift-by-1. Bit-exact vs the serial reference because each
 * a[i+1] is read (original) before it is overwritten. Split into per-thread
 * blocks; the block boundary value is captured in bd[] up front so no cross-
 * thread read/write race on the shared a[j] element remains. */

static inline void block512(double *a, const double *b, int64_t lo, int64_t hi, double bound) {
  int64_t j = lo;
  for (; j + 8 <= hi - 1; j += 8) {
    __m512d va = _mm512_loadu_pd(a + j + 1);   /* a_in[j+1..j+8] (in-block) */
    __m512d vb = _mm512_loadu_pd(b + j);       /* b[j..j+7] */
    _mm512_storeu_pd(a + j, _mm512_add_pd(va, vb));
  }
  for (; j < hi - 1; ++j) a[j] = a[j+1] + b[j];
  a[hi-1] = bound + b[hi-1];
}
static inline void block256(double *a, const double *b, int64_t lo, int64_t hi, double bound) {
  int64_t j = lo;
  for (; j + 4 <= hi - 1; j += 4) {
    __m256d va = _mm256_loadu_pd(a + j + 1);   /* a_in[j+1..j+4] */
    __m256d vb = _mm256_loadu_pd(b + j);       /* b[j..j+3] */
    _mm256_storeu_pd(a + j, _mm256_add_pd(va, vb));
  }
  for (; j < hi - 1; ++j) a[j] = a[j+1] + b[j];
  a[hi-1] = bound + b[hi-1];
}

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  int64_t N = LEN_1D;
  if (N <= 1) return;
  int64_t M = N - 1;
  int nt = omp_get_max_threads();
  if (nt < 1) nt = 1;
  int64_t c = (M + nt - 1) / nt;
  if (c < 1) c = 1;
  int nb = (int)((M + c - 1) / c);
  double bd[2048];
  if (nb + 1 > 2048) nb = 2047;
  c = (M + nb - 1) / nb;
  for (int k = 0; k < nb; ++k) bd[k] = a[(int64_t)k * c];
  bd[nb] = a[N-1];
  int use512 = __builtin_cpu_supports("avx512f");
  #pragma omp parallel for schedule(static)
  for (int k = 0; k < nb; ++k) {
    int64_t lo = (int64_t)k * c;
    int64_t hi = lo + c; if (hi > M) hi = M;
    if (use512) block512(a, b, lo, hi, bd[k+1]);
    else        block256(a, b, lo, hi, bd[k+1]);
  }
}
