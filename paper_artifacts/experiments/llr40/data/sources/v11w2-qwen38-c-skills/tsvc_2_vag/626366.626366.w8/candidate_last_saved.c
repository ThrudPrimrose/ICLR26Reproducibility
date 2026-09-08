/* TSVC tsvc_2 "vag" random-permutation gather: a[i] = b[ip[i]].
 *
 * ip is a random permutation of [0, n): every 8-byte read of b is a random
 * DRAM line, so the kernel is memory-bound.  Strategy:
 *   - one contiguous span per thread (static split), no load imbalance,
 *   - 8 elements per iteration, each via the 4-lane int32->double gather
 *     (_mm256_i32gather_pd is only guaranteed to fill its low 128 bits on
 *     this toolchain, so two gathers + insertf128 build the 64B line),
 *   - non-temporal 64B stores for a (written once, never re-read here),
 *     which skips the read-for-ownership traffic.
 */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D)
{
  const int64_t n = LEN_1D;
  if (n <= 0) return;
  const int nt = omp_get_max_threads();

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int64_t per = (n + nt - 1) / nt;
    int64_t lo = (int64_t)tid * per;
    int64_t hi = lo + per;
    if (hi > n) hi = n;
    int64_t i = lo;

    /* align this span's start to the 64B boundary the NT store needs */
    while (i < hi && ((uintptr_t)(a + i) & 63u) != 0) {
      a[i] = b[ip[i]];
      i++;
    }

    const int64_t vlim = hi - (hi % 8);
    for (; i < vlim; i += 8) {
      __m128i i0 = _mm_loadu_si128((const __m128i *)(ip + i));
      __m128i i1 = _mm_loadu_si128((const __m128i *)(ip + i + 4));
      __m256d v0 = _mm256_i32gather_pd(b, i0, 8); /* low 128 valid */
      __m256d v1 = _mm256_i32gather_pd(b, i1, 8); /* low 128 valid */
      __m256d v = _mm256_insertf128_pd(v0, _mm256_castpd256_pd128(v1), 1);
      _mm256_stream_pd(a + i, v);
    }
    for (; i < hi; i++) a[i] = b[ip[i]];
  }
}
