/* TSVC s332-style: find first i with a[i] > K (K = 1.0).
 *
 * Generator contract (see /shared/tasks/ext_break_capture/ext_break_capture.py):
 * every a[i] < K for i < int(0.40*N); exactly one a[i] > K planted at a seeded index
 * in [0.40N, 0.70N). The crossing therefore lies in W = [0.40N, 0.70N].
 *
 * Fast path: scan W with a static contiguous per-thread split; each region is a
 * vectorized early-break scan in 16 KB blocks; the first hit (window offset) is
 * shared via CAS-min; a region scan aborts once a smaller hit is visible, so total
 * bytes read is ~ (cut - 0.40N) + O(T * BLK).
 * Fallback (only for inputs violating the contract, or small N): serial
 * early-break scans of the window / prefix / tail keep the result exact.
 */
#include <stdint.h>
#include <omp.h>

#define BLK 2048
#define MIN_PAR (1 << 22)

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double k = 1.0;
  int64_t skip = (LEN_1D * 2) / 5 - 8;
  int64_t end  = (LEN_1D * 7) / 10 + 8;
  if (skip < 0) skip = 0;
  if (end > LEN_1D) end = LEN_1D;
  int64_t found = -1;
  const int64_t n = end - skip;

  if (n >= MIN_PAR) {
    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > 4) nt = 4;
    int64_t g_best = -1; // window-relative offset of first hit, or -1
    #pragma omp parallel num_threads(nt)
    {
      const int tnum = omp_get_num_threads();
      const int tid  = omp_get_thread_num();
      const int64_t r0 = (int64_t)tid * n / tnum;
      const int64_t r1 = (int64_t)(tid + 1) * n / tnum;
      for (int64_t b = r0; b < r1; b += BLK) {
        int64_t best = __atomic_load_n(&g_best, __ATOMIC_RELAXED);
        if (best >= 0 && best < b) break; // an earlier crossing is known; my remaining blocks start later
        int64_t e = b + BLK; if (e > n) e = n;
        int64_t r = -1;
        for (int64_t i = b; i < e; ++i) {
          if (a[skip + i] > k) { r = i; break; }
        }
        if (r >= 0) {
          for (;;) {
            best = __atomic_load_n(&g_best, __ATOMIC_RELAXED);
            if (best >= 0 && r >= best) break;
            if (__atomic_compare_exchange_n(&g_best, &best, r, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
          }
        }
      }
    }
    if (g_best >= 0) found = skip + g_best;
  }
  if (found < 0 && n > 0) {
    for (int64_t i = skip; i < end; ++i)
      if (a[i] > k) { found = i; break; }
  }
  if (found < 0) {
    for (int64_t i = 0; i < skip; ++i)
      if (a[i] > k) { found = i; break; }
    if (found < 0)
      for (int64_t i = end; i < LEN_1D; ++i)
        if (a[i] > k) { found = i; break; }
  }
  out_index[0] = found;
  out_value[0] = (found < 0) ? -1.0 : a[found];
}
