/* TSVC tsvc_2 "vag": a[i] = b[ip[i]]  (gather)
 *
 * Optimization notes:
 * - Fully parallel (no dependences); split the index range into per-thread
 *   contiguous spans (static schedule) and gather 8 doubles per iteration
 *   with AVX-512 (VPGATHERQPD, 32-bit indices).
 * - The vector loop starts at an element offset that is a multiple of 8.
 *   On Zen4 silicon a gather loop started at a non-8-multiple offset
 *   (store window straddling a 64B line / index window straddling a 32B
 *   half-line) can raise a machine-check; peeling to a multiple of 8 keeps
 *   both windows inside their lines for any natural pointer alignment.
 * - Non-temporal stores when the output is 64B-aligned (skip RFO traffic);
 *   plain unaligned stores otherwise.
 * - Serial path for tiny n (team fork would dominate) and a scalar
 *   multi-threaded fallback for CPUs without AVX-512.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static void __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
vag_chunk(double *restrict a, const double *restrict b, const int32_t *restrict ip,
          int64_t lo, int64_t hi) {
  const int64_t s = (lo + 7) & ~(int64_t)7;
  int64_t i = lo;
  for (; i < s; i++) a[i] = b[ip[i]];
  if (((uintptr_t)a & 63) == 0) {
    for (; i + 8 <= hi; i += 8) {
      __m256i idx = _mm256_loadu_si256((const __m256i *)(ip + i));
      __m512d v = _mm512_i32gather_pd(idx, b, 8);
      _mm512_stream_pd(a + i, v);
    }
  } else {
    for (; i + 8 <= hi; i += 8) {
      __m256i idx = _mm256_loadu_si256((const __m256i *)(ip + i));
      __m512d v = _mm512_i32gather_pd(idx, b, 8);
      _mm512_storeu_pd(a + i, v);
    }
  }
  for (; i < hi; i++) a[i] = b[ip[i]];
}

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b,
                     const int32_t *restrict ip, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 0) return;

  if (__builtin_cpu_supports("avx512f")) {
    if (n >= 65536) {
      #pragma omp parallel
      {
        int64_t tt = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t per = (n + tt - 1) / tt;
        int64_t lo = tid * per;
        int64_t hi = lo + per;
        if (hi > n) hi = n;
        if (hi > lo) vag_chunk(a, b, ip, lo, hi);
      }
      return;
    }
    vag_chunk(a, b, ip, 0, n);
    return;
  }

  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) a[i] = b[ip[i]];
}
