/* TSVC tsvc_2_5 quasi_affine_reduce_odd: out[0] = sum(a[i] for i in 1,3,5,...) */
#include <stdint.h>
#include <immintrin.h>

#if defined(__AVX512F__)

/* shuffle [lo,hi] -> [lo[1],lo[3],lo[5],lo[7],hi[1],hi[3],hi[5],hi[7]] */
#define ODD_PICK_IMM 0x35B1

static inline __attribute__((always_inline)) double
reduce4(__m512d a0, __m512d a1, __m512d a2, __m512d a3) {
  __m512d s0 = _mm512_add_pd(a0, a1);
  __m512d s1 = _mm512_add_pd(a2, a3);
  s0 = _mm512_add_pd(s0, s1);
  s0 = _mm512_add_pd(s0, _mm512_permutex_pd(s0, 0xD8));
  s0 = _mm512_add_pd(s0, _mm512_permutex_pd(s0, 0x32));
  s0 = _mm512_add_pd(s0, _mm512_permute_pd(s0, 1));
  double r[8];
  _mm512_storeu_pd(r, s0);
  return r[0];
}

#endif /* __AVX512F__ */

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out,
                                  const int64_t LEN_1D) {
  double total = 0.0;
  if (LEN_1D < 2) {
    out[0] = total;
    return;
  }

#if defined(__AVX512F__)
  const double *p = a;
  const int64_t n16 = LEN_1D / 16;      /* full 16-double chunks       */
  const int64_t n   = LEN_1D / 2;       /* number of odd-indexed terms */
  __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
  __m512d acc2 = _mm512_setzero_pd(), acc3 = _mm512_setzero_pd();
  int64_t k = 0;
  for (; k + 4 <= n16; k += 4, p += 64) {
    __m512d lo0 = _mm512_loadu_pd(p);
    __m512d hi0 = _mm512_loadu_pd(p + 8);
    __m512d lo1 = _mm512_loadu_pd(p + 16);
    __m512d hi1 = _mm512_loadu_pd(p + 24);
    __m512d lo2 = _mm512_loadu_pd(p + 32);
    __m512d hi2 = _mm512_loadu_pd(p + 40);
    __m512d lo3 = _mm512_loadu_pd(p + 48);
    __m512d hi3 = _mm512_loadu_pd(p + 56);
    acc0 = _mm512_add_pd(acc0, _mm512_shuffle_pd(lo0, hi0, ODD_PICK_IMM));
    acc1 = _mm512_add_pd(acc1, _mm512_shuffle_pd(lo1, hi1, ODD_PICK_IMM));
    acc2 = _mm512_add_pd(acc2, _mm512_shuffle_pd(lo2, hi2, ODD_PICK_IMM));
    acc3 = _mm512_add_pd(acc3, _mm512_shuffle_pd(lo3, hi3, ODD_PICK_IMM));
  }
  for (; k < n16; k++, p += 16) {
    __m512d lo = _mm512_loadu_pd(p);
    __m512d hi = _mm512_loadu_pd(p + 8);
    acc0 = _mm512_add_pd(acc0, _mm512_shuffle_pd(lo, hi, ODD_PICK_IMM));
  }
  total = reduce4(acc0, acc1, acc2, acc3);

  {
    const int64_t base = 16 * n16;
    const int64_t r = n - 8 * n16;      /* leftover odd terms: 0..7 */
    for (int64_t j = 0; j < r; j++) {
      total += a[base + 1 + 2 * j];
    }
  }
#else
  for (int64_t i = 1; i < LEN_1D; i += 2) {
    total += a[i];
  }
#endif
  out[0] = total;
}
