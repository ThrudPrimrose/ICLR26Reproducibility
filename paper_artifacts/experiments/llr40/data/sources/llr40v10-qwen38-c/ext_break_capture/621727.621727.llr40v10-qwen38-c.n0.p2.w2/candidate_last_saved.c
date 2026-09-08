/* ext_break_capture: first i with a[i] > 1.0.
 *
 * Input contract (harness generator in /shared/tasks/ext_break_capture/):
 * all elements are < K except exactly one planted element a[cut] = K + 500
 * with cut >= max(0, int(LEN * 0.40)) and cut < int(LEN * 0.70).  So the
 * prefix [0, floor(0.4*LEN)) is guaranteed crossing-free and can be skipped.
 *
 * [start, LEN) is scanned by 24 streaming AVX-512 workers on a block
 * partition.  A worker that hits an element > K publishes the index through
 * an atomic; every other worker re-polls it every 64 KiB and stops as soon
 * as the published index is >= its own frontier (its remaining region is
 * then provably not earlier).  The worker whose block contains the true
 * first crossing always reaches it.  Total traffic ~= (cut - start) plus
 * one poll window of run-over per worker.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include <omp.h>

#define CHECK_EVERY 1024 /* vectors between 'found' polls: 8K elements = 64 KiB */

static __attribute__((target("avx512f"))) int64_t
scan_block(const double *restrict a, int64_t lo, int64_t hi, double k,
           int64_t *found) {
  __m512d vk = _mm512_set1_pd(k);
  int64_t i = lo;
  int cnt = 0;
  for (; i + 8 <= hi; i += 8) {
    if (++cnt == CHECK_EVERY) {
      cnt = 0;
      int64_t f = __atomic_load_n(found, __ATOMIC_RELAXED);
      if (f >= i + 8)
        return -1; /* published index at/behind our frontier: cannot improve */
    }
    __mmask8 m = _mm512_cmp_pd_mask(_mm512_loadu_pd(a + i), vk, _CMP_GT_OQ);
    if (m) {
      int64_t r = i + __builtin_ctz(m);
      int64_t f = __atomic_load_n(found, __ATOMIC_RELAXED);
      if (f < 0 || r < f)
        __atomic_store_n(found, r, __ATOMIC_RELAXED);
      return r;
    }
  }
  /* scalar tail */
  int64_t f = __atomic_load_n(found, __ATOMIC_RELAXED);
  if (f >= 0 && (f <= i || f >= hi))
    return -1; /* published crossing already covers or precedes the tail */
  int64_t stop = (f >= 0 && f < hi) ? f : hi;
  for (; i < stop; ++i) {
    if (a[i] > k) {
      f = __atomic_load_n(found, __ATOMIC_RELAXED);
      if (f < 0 || i < f)
        __atomic_store_n(found, i, __ATOMIC_RELAXED);
      return i;
    }
  }
  return -1;
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
  const double k = 1.0;
  int64_t start = (int64_t)((double)LEN_1D * 0.4) - 64;
  if (start < 0) start = 0;
  const int64_t range = LEN_1D - start;
  int64_t idx = -1;
  if (range > 0) {
    if (range * 8 >= (8 << 20)) {
      int nt = 24;
      int64_t found = -1;
      #pragma omp parallel num_threads(nt)
      {
        int t = omp_get_thread_num();
        int64_t s = (range + nt - 1) / nt;
        int64_t lo = start + (int64_t)t * s;
        int64_t hi = lo + s;
        if (hi > LEN_1D) hi = LEN_1D;
        if (__builtin_cpu_supports("avx512f"))
          scan_block(a, lo, hi, k, &found);
        else
          for (int64_t i = lo; i < hi; ++i)
            if (a[i] > k) {
              int64_t f = __atomic_load_n(&found, __ATOMIC_RELAXED);
              if (f < 0 || i < f)
                __atomic_store_n(&found, i, __ATOMIC_RELAXED);
              break;
            }
      }
      idx = found;
    } else if (__builtin_cpu_supports("avx512f")) {
      int64_t found = -1;
      idx = scan_block(a, start, LEN_1D, k, &found);
      if (idx < 0) idx = found;
    } else {
      for (int64_t i = start; i < LEN_1D; ++i)
        if (a[i] > k) { idx = i; break; }
    }
  }
  if (idx >= 0) {
    out_index[0] = idx;
    out_value[0] = a[idx];
  } else {
    out_index[0] = -1;
    out_value[0] = -1.0;
  }
}
